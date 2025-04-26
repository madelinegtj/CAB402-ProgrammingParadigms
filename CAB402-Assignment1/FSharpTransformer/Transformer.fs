module Transformer

open Types
open IO
open HelperFunctions
open Attention

// Return a new function that takes the head number, requested position, and position within head, returning the key/value
// from [head number][postion within head] in either the newValues matrix if the token position is the current position,
// otherwise in the history/cache table at [requested position][layer].
let createLookupFunction (previousValues:MultiHead[][]) (newValues:MultiHead) (tokenPosition:int) (layer:int): (int -> int -> int -> float) =
    fun headNumber requestedPosition positionWithinHead ->
        
        // If the requested position = the current token position
        if requestedPosition = tokenPosition then
            // return key/value from freshly computed newValues
            newValues.[headNumber].[positionWithinHead]
        else
            // Otherwise, return from previously cached keys/values
            previousValues.[requestedPosition].[layer].[headNumber].[positionWithinHead]

// Processes one layer of the transformer model. This is equivalent to the first for loop in the C# transformer() function.
// The parameters you will need are stored in the model.weights array under index layer.
// You need to:
// - Apply layer normalization to the current vector before attention using normalizeInputWeights
// - Generate query, key and value vectors by multiplying the current vector by the corresponding query (wq), key (qk) and value (wv)
//   matrices for this layer. You will need to use the reshapeToMultipleHeads() function to split these vectors.
// - Apply Rotary Position Embedding(RoPE) to query and key vectors. The value vector is not rotated.
// - Use the attention function to compute multi-head attention for the query/key/value vectors.
// - Project concatenated attention outputs with the output matrix (wo) to produce final attention.
// - Add the residual connection (input vector).
// - Apply layer normalization before the final feed-forward neural network (normalizeAttentionWeights).
// - Feed-forward network component: Matrix multiply w1 and w3, sigmoid is only applied to w1.
// - Then the product of these two matrices is multiplied by w2 with second residual connection.
let feedforwardOneLayer (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input: Vector) (layer: int) : Vector * MultiHead * MultiHead =
    
    // Normalize input before attention
    let layerWeights = model.weights.[layer]
    let normalizedInput = rootMeanSquareNormalize layerWeights.normalizeInputWeights input

    // Create query (q), key (k), and value (v) vectors
    let q = matrixMultiply layerWeights.wq normalizedInput
    let k = matrixMultiply layerWeights.wk normalizedInput
    let v = matrixMultiply layerWeights.wv normalizedInput
    
    // Split q, k, v into multiple heads
    let headSize = model.headSize
    let queryHeads = reshapeToMultipleHeads headSize q
    let keyHeads = reshapeToMultipleHeads headSize k
    let valueHeads = reshapeToMultipleHeads headSize v
    
    // Apply Rotary Position Embedding to query and key
    let rotatedQueryHeads = rotateVector model.rotationCoefficients.[tokenPosition] queryHeads
    let rotatedKeyHeads = rotateVector model.rotationCoefficients.[tokenPosition] keyHeads
    
    // Create lookup functions for key and value vectors
    let keyLookup = createLookupFunction keyCache rotatedKeyHeads tokenPosition layer
    let valueLookup = createLookupFunction valueCache valueHeads tokenPosition layer
    // Compute multi-head attention output
    let attentionOutput = attention keyLookup valueLookup tokenPosition rotatedQueryHeads

    // Project the attention output back to model dimension and add the first residual connection
    let projected = matrixMultiply layerWeights.wo attentionOutput
    let residual1 = add input projected
    // Normalize after residual connection
    let normalizedResidual = rootMeanSquareNormalize layerWeights.normalizeAttentionWeights residual1
    
    // Feed-forward network with gating
    let hidden1 = matrixMultiply layerWeights.w1 normalizedResidual
    let hidden3 = matrixMultiply layerWeights.w3 normalizedResidual
    let activated = sigmoidActivation hidden1
    let gated = elementWiseMultiply activated hidden3
    let ffnOutput = matrixMultiply layerWeights.w2 gated

    // Add second residual connection
    let residual2 = add residual1 ffnOutput

    (residual2, rotatedKeyHeads, valueHeads)


// Returns a new array with the newElement added to array.
let appendElement (array: 'T[]) (newElement: 'T) : 'T[] =
    Array.append array [| newElement |]

// Feeds an input vector through all layers of the transformer.
// This function is also responsible for updating the key/value cache that is used to retrieve the vectors in later layers.
// The cache is "updated" for each layer by appending to the end of the array representing the cache.
let feedForwardAllLayers (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input:Vector)  : Vector * MultiHead[] * MultiHead[] =
    // Use List.fold to process each layer, accumulating the input with each.
    let Folder (input, previousKeys, previousValues) layer =
        let (output, keys, values) = feedforwardOneLayer model keyCache valueCache tokenPosition input layer
        (output, appendElement previousKeys keys,  appendElement previousValues values)
    List.fold Folder (input, Array.empty, Array.empty) [0 .. model.numberOfLayers-1]


// Uses all the transformer model's layers to predict the next token that follows token.
// The output is the logits for each token, and the key/value cache for all layers for this token.
// This function roughly equates to the first copy() call and final rmsnorm()/matmul() calls in the C# transformer() method.
let feedForward (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (token:Token) : Vector * MultiHead[] * MultiHead[] =
    
    // Embed the token into a vector
    let embedding = model.tokenEmbedding.[token]
    // Pass the embedding through all transformer layers
    let (output, keys, values) = feedForwardAllLayers model keyCache valueCache tokenPosition embedding

    // Normalize the final output vector
    let normalized = rootMeanSquareNormalize model.normalizeOutputWeights output
    // Project the normalized vector back into the vocabulary space to get logits
    let logits = matrixMultiply model.tokenEmbedding normalized

    (logits, keys, values)

// Obtains the logits for the next token, and selects the token to return based on the provided decoder function.
// You should also return the updated key/value cache.
let generateNextToken (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][])  (tokenPosition:int) (token:Token) (decoder:Vector->Token) : Token * MultiHead[] * MultiHead[] =
    
    // Feed token through model to obtain logits and updated caches
    let (logits, keys, values) = feedForward model keyCache valueCache tokenPosition token
    // Decode the logits to select the next token
    let nextToken = decoder logits

    (nextToken, keys, values)

// Generates a sequence of tokens using the specified decoder.
// This function is responsible for appending the cache of key/values for all layers to the "main" key/value cache,
// which contains the key/values for all layers of every preceding token.
// The start and end of the sequence are indicated by the token 1, therefore we should stop producing tokens after
// a token of 1 is predicted. Each token is also printed out as it is generated.
let generateTokenSequence (model: Model) (decoder:Vector->Token) : string seq = 
    (1, 0, Array.empty, Array.empty) 
    |> Seq.unfold (fun (token, tokenPosition, previousKeys, previousValues) -> 
        let (nextToken, keys, values) = generateNextToken model previousKeys previousValues tokenPosition token decoder
        let newKeys = appendElement previousKeys keys
        let newValues = appendElement previousValues values
        if nextToken = 1 || tokenPosition+1 = model.seqenceLength
        then None
        else
            Some (model.vocabulary.[nextToken], (nextToken, tokenPosition+1, newKeys, newValues)))

let tellStory (model: Model) (decoder:Vector->Token) : unit =
    generateTokenSequence model decoder
    |> printStory