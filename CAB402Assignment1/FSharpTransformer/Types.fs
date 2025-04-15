module Types

open System.Numerics

/// Represents a one-dimensional vector of 32-bit floating point elements.
type Vector = float[]

/// Represents one "head" of multi-head attention. Different heads focus on different parts
/// and characteristics of the input.
type MultiHead = Vector[]

/// Represents the vector that is input to layer normalization.
type WeightVector = float[]

/// Learnt weight matrix used in the transformer model.
type WeightMatrix = float[][]

/// Tokens are represented by integers, with each integer corresponding to a word or part of a word.
type Token = int

type Weights = {
    normalizeInputWeights: Vector /// Layer normalization applied before multi-head attention.
    normalizeAttentionWeights: Vector /// Layer normalization applied before feed-forward.
    wq: WeightMatrix /// Learnt matrix wk used to compute query vector for a given layer.
    wk: WeightMatrix /// Learnt matrix wk used to compute value vector for a given layer.
    wv: WeightMatrix /// Learnt matrix wk used to compute key vector for a given layer.
    wo: WeightMatrix /// Learnt matrix wo applied after multi-head attention.
    w1: WeightMatrix /// Learnt matrix w1 used in feed-forward stage.
    w2: WeightMatrix /// Learnt matrix w2 used in feed-forward stage.
    w3: WeightMatrix /// Learnt matrix w3 used in feed-forward stage.
}

type Model = {
    headSize: int /// Size of each "head" in multi-head attention.
    numberOfHeads: int /// Total number of heads to compute attention for.
    numberOfLayers: int /// Total number of layers in the transformer model.
    vocabularySize: int /// Number of possible tokens.
    seqenceLength: int /// Maximum number of tokens the model can output in a single sequence.
    tokenEmbedding: WeightMatrix /// Learnt embedding vector for each token in the vocabulary.
    weights: Weights[] /// Learnt model parameters, see above.
    normalizeOutputWeights: Vector /// Layer normalization applied as final stage.
    rotationCoefficients: Complex[][] /// Lookup table for Rotary Positional Encoding (RoPE).
    vocabulary: string[] /// Strings containing the word for each token in the vocabulary.
}