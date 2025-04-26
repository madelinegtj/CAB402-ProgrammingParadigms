module Decoders

open Types

// Selects the most likely token based on the probabilities returned for each.
// This is effectively argmax(logits).
let mostLikely (vector:Vector) : Token = 
    vector
    |> Array.mapi (fun i x -> (i, x)) 
    |> Array.maxBy snd 
    |> fst

// Selects a random token from the most k likely tokens.
// This makes the output more "human-like" by not returning the exact same sequence
// every time, however it is more difficult to test due to being non-deterministic.
let topKSample k (vector:Vector) : Token =
    let sortedIndices = vector |> Array.mapi (fun i x -> (i, x)) |> Array.sortByDescending snd |> Array.take k |> Array.map fst
    let topKValues = sortedIndices |> Array.map (fun i -> vector.[i])
    let cumulativeSum = Array.scan (+) 0.0 topKValues
    let totalSum = cumulativeSum.[cumulativeSum.Length - 1]
    let randomValue = System.Random().NextDouble() * totalSum
    let chosenIndex = Array.findIndex (fun x -> x >= randomValue) cumulativeSum
    sortedIndices.[chosenIndex-1]
