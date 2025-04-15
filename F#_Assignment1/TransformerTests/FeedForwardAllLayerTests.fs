namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open ReadResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardAllLayerTests () =

    let model53M = readModel53M

    let keys1 = read3D "keys1"
    let values1 = read3D "values1"
    let keys2 = read3D "keys2"
    let values2 = read3D "values2"
    let keys3 = read3D "keys3"
    let values3 = read3D "values3"
    let expected1 = read2D "output1"
    let expected2 = read2D "output2"
    let expected3 = read2D "output3"

    [<TestMethod>]
    member this.Test1 () =
        let input1 = model53M.tokenEmbedding[1]
        let (output1, keyCache1, valueCache1) = feedForwardAllLayers model53M Array.empty Array.empty 0 input1
        compare1D "output" expected1.[5] output1
        compare3D "keyCache" keys1 keyCache1
        compare3D "valueCache" values1 valueCache1


    [<TestMethod>]
    member this.Test2 () =
        let input2 = model53M.tokenEmbedding[9038]
        let (output2, keyCache2, valueCache2) = feedForwardAllLayers model53M [| keys1|] [|values1|] 1 input2
        compare1D "output" expected2.[5] output2
        compare3D "keyCache" keys2 keyCache2
        compare3D "valueCache" values2 valueCache2
  

    [<TestMethod>]
    member this.Test3 () =
        let input3 = model53M.tokenEmbedding[2501]
        let (output3, keyCache3, valueCache3) = feedForwardAllLayers model53M [| keys1; keys2|] [|values1; values2 |] 2 input3
        compare1D "output" expected3.[5] output3
        compare3D "keyCache" keys3 keyCache3
        compare3D "valueCache" values3 valueCache3
