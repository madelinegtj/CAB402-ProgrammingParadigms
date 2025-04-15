namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open ReadResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardTests () =

    let model53M = readModel53M

    let keys1 = read3D "keys1"
    let values1 = read3D "values1"
    let keys2 = read3D "keys2"
    let values2 = read3D "values2"
    let keys3 = read3D "keys3"
    let values3 = read3D "values3"
    let logits1 = read1D "logits1"
    let logits2 = read1D "logits2"
    let logits3 = read1D "logits3"

    [<TestMethod>]
    member this.Test1 () =
        let (output1, keyCache1, valueCache1) = feedForward model53M Array.empty Array.empty 0 1
        compare1D "logits" logits1 output1
        compare3D "keyCache" keys1 keyCache1
        compare3D "valueCache" values1 valueCache1

    [<TestMethod>]
    member this.Test2 () =
        let (output2, keyCache2, valueCache2) = feedForward model53M [| keys1|] [|values1|] 1 9038
        compare1D "logits" logits2 output2
        compare3D "keyCache" keys2 keyCache2
        compare3D "valueCache" values2 valueCache2

    [<TestMethod>]
    member this.Test3 () =
        let (output3, keyCache3, valueCache3) = feedForward model53M [| keys1; keys2|] [|values1; values2 |] 2 2501
        compare1D "logits" logits3 output3
        compare3D "keyCache" keys3 keyCache3
        compare3D "valueCache" values3 valueCache3