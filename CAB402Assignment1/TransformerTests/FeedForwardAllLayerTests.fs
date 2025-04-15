namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open SaveResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardAllLayerTests () =

    let model53M = readModel53M

    let input = model53M.tokenEmbedding[1]

    [<TestMethod>]
    member this.Test1 () =
        let (output, keyCache, valueCache) = feedForwardAllLayers model53M Array.empty Array.empty 0 input
        saveResults3 "test1b" output keyCache valueCache
        compareResults3D "test1b" output keyCache valueCache
    [<TestMethod>]
    member this.Test2 () =
        let (output1, keyCache1, valueCache1) = feedForwardAllLayers model53M Array.empty Array.empty 0 input
        let input2 = model53M.tokenEmbedding[9038]
        let (output2, keyCache2, valueCache2) = feedForwardAllLayers model53M [| keyCache1|] [|valueCache1|] 1 input2
        saveResults3 "test2b" output2 keyCache2 valueCache2
        compareResults3D "test2b" output2 keyCache2 valueCache2
    [<TestMethod>]
    member this.Test3 () =
        let (output1, keyCache1, valueCache1) = feedForwardAllLayers model53M Array.empty Array.empty 0 input
        let input2 = model53M.tokenEmbedding[9038]
        let (output2, keyCache2, valueCache2) = feedForwardAllLayers model53M [| keyCache1|] [|valueCache1|] 1 input2
        let input3 = model53M.tokenEmbedding[2501]
        let (output3, keyCache3, valueCache3) = feedForwardAllLayers model53M [| keyCache1; keyCache2|] [|valueCache1; keyCache2 |] 2 input3
        saveResults3 "test3b" output3 keyCache3 valueCache3
        compareResults3D "test3b" output3 keyCache3 valueCache3