namespace TransformerFunctions

open Transformer
open Decoders
open Microsoft.VisualStudio.TestTools.UnitTesting
open SaveResults
open TransformerTestsHelpers

[<TestClass>]
type GenerateNextTokenTests () =

    let model53M = readModel53M

    [<TestMethod>]
    member this.Test1 () =
        let (nextToken, keyCache, valueCache) = generateNextToken model53M Array.empty Array.empty 0 1 mostLikely
        Assert.AreEqual<int>(9038, nextToken)
        compareCache "test1c"  keyCache  valueCache

    [<TestMethod>]
    member this.Test2 () =
        let (nextToken1, keyCache1, valueCache1) = generateNextToken model53M Array.empty Array.empty 0 1 mostLikely
        let (nextToken2, keyCache2, valueCache2) = generateNextToken model53M [| keyCache1|] [|valueCache1|] 1 nextToken1 mostLikely
        Assert.AreEqual<int>(2501, nextToken2)
        compareCache "test2c" keyCache2 valueCache2

    [<TestMethod>]
    member this.Test3 () =
        let (nextToken1, keyCache1, valueCache1) = generateNextToken model53M Array.empty Array.empty 0 1 mostLikely
        let (nextToken2, keyCache2, valueCache2) = generateNextToken model53M [| keyCache1|] [|valueCache1|] 1 nextToken1 mostLikely
        let (nextToken3, keyCache3, valueCache3) = generateNextToken model53M [| keyCache1; keyCache2|] [|valueCache1; keyCache2 |] 2 nextToken2 mostLikely
        Assert.AreEqual<int>(263, nextToken3)
        compareCache "test3c" keyCache3 valueCache3