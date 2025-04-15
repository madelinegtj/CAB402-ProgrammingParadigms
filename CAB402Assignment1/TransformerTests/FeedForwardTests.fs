namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open SaveResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardTests () =

    let model53M = readModel53M

    [<TestMethod>]
    member this.Test1 () =
        let (output, keyCache, valueCache) = feedForward model53M Array.empty Array.empty 0 1
        saveResults3 "test1c" output keyCache valueCache
        compareResults3D "test1c" output keyCache valueCache

    [<TestMethod>]
    member this.Test2 () =
        let (output1, keyCache1, valueCache1) = feedForward model53M Array.empty Array.empty 0 1
        let (output2, keyCache2, valueCache2) = feedForward model53M [| keyCache1|] [|valueCache1|] 1 9038
        saveResults3 "test2c" output2 keyCache2 valueCache2
        compareResults3D "test2c" output2 keyCache2 valueCache2

    [<TestMethod>]
    member this.Test3 () =
        let (output1, keyCache1, valueCache1) = feedForward model53M Array.empty Array.empty 0 1
        let (output2, keyCache2, valueCache2) = feedForward model53M [| keyCache1|] [|valueCache1|] 1 9038
        let (output3, keyCache3, valueCache3) = feedForward model53M [| keyCache1; keyCache2|] [|valueCache1; keyCache2 |] 2 2501
        saveResults3 "test3c" output3 keyCache3 valueCache3
        compareResults3D "test3c" output3 keyCache3 valueCache3