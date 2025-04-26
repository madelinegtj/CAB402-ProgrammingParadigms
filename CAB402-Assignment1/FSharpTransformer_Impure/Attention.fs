module Attention

open Types
open HelperFunctions


// Computes attention for a single token.
// Equivalent to the innermost loop of transformer() in the C# implementation.
// i.e. compute the square root of the dot product between query and key vectors.
// Hint: Use the keyLookup function, as we do not have the key vector directly here.
let attentionScore (query:Vector) (keyLookup:int->float) : float =    
    
    // For each element q in the query vector, multiply it by the corresponding key value,
    // then sum all the multiplied terms to calculate the dot product
    let dotProduct = Array.mapi (fun i q -> q * keyLookup i) query |> Array.sum

    // Scale the dot product values
    dotProduct / sqrt (float query.Length)      

// Compute the dot product of the attention vector with the value vector.
let weightedAttention (attention: Vector) (valueLookup:int->float) : float =
    
    // For each attention score a at position i, multiply it by the corresponding value at position i
    // then sum all the products to get the final weighted sum
    Array.mapi (fun i a -> a * valueLookup i) attention |> Array.sum

// Computes attention for one head of multi-head attention, using the query, key and value vectors.
// This is equivalent to the n_heads loop in the transformer() function in the C# implementation.    
let attentionForOneHead (keyLookup:int->int->float) (valueLookup:int->int->float) (tokenPosition:int) (query: Vector): Vector =
    
    // For each previous token position, compute the attention score between the query and its corresponding key vector
    let scores = Array.init (tokenPosition + 1) (fun pos ->
        let keyForPos i = keyLookup pos i
        attentionScore query keyForPos
    )
    // Normalize scores into attention weights using softmax
    let weights = softMax scores

    // For each output dimension, compute weighted sum of value vectors
    Array.init query.Length (fun i ->
        Array.mapi (fun pos weight -> weight * valueLookup pos i) weights |> Array.sum
    )

// Computes attention for all heads in multi-head attention.
// Hint: Instead of returning multiple vectors, one for each head, this array should be flattened with flattenMultipleHeads().
let attention (keyLookup:int->int->int->float) (valueLookup:int->int->int->float) (tokenPosition:int) (query: MultiHead) : Vector =
    
    // A function to process one head
    let processHead head headQuery =
        // Find the Key lookup and Value lookup for this head
        let headKeyLookup = (fun pos i -> keyLookup head pos i)    
        let headValueLookup = (fun pos i -> valueLookup head pos i)
        // Compute the attention output for this head
        attentionForOneHead headKeyLookup headValueLookup tokenPosition headQuery
    
    // Process all heads
    let headResults =
        query
        |> Array.mapi (fun head headQuery -> processHead head headQuery)
     
    // Flatten the results across all heads into a single vector 
    flattenMultipleHeads headResults
