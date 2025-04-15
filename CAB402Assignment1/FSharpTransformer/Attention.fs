module Attention

open Types
open HelperFunctions

// Computes attention for a single token.
// Equivalent to the innermost loop of transformer() in the C# implementation.
// i.e. compute the square root of the dot product between query and key vectors.
// Hint: Use the keyLookup function, as we do not have the key vector directly here.
let attentionScore (query:Vector) (keyLookup:int->float) : float =    
    // TODO: Implement this function.
    raise (System.NotImplementedException("Attention attentionScore not implemented"))

// Compute the dot product of the attention vector with the value vector.
let weightedAttention (attention: Vector) (valueLookup:int->float) : float =
    // TODO: Implement this function.
    raise (System.NotImplementedException("Attention weightedAttention not implemented"))

// Computes attention for one head of multi-head attention, using the query, key and value vectors.
// This is equivalent to the n_heads loop in the transformer() function in the C# implementation.    
let attentionForOneHead (keyLookup:int->int->float) (valueLookup:int->int->float) (tokenPosition:int) (query: Vector): Vector =
    // TODO: Implement this function.
    raise (System.NotImplementedException("Attention attentionForOneHead not implemented"))

// Computes attention for all heads in multi-head attention.
// Hint: Instead of returning multiple vectors, one for each head, this array should be flattened with flattenMultipleHeads().
let attention  (keyLookup:int->int->int->float) (valueLookup:int->int->int->float) (tokenPosition:int) (query: MultiHead) : Vector =
    // TODO: Implement this function.
    raise (System.NotImplementedException("Attention attention not implemented"))
