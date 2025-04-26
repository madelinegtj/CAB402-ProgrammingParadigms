namespace TransformerFunctions

open Transformer
open Decoders
open Microsoft.VisualStudio.TestTools.UnitTesting
open ReadResults
open TransformerTestsHelpers

[<TestClass>]
type GenerateNextTokenTests () =

    let model53M = readModel53M

    let keys1 = read3D "keys1"
    let values1 = read3D "values1"
    let keys2 = read3D "keys2"
    let values2 = read3D "values2"
    let keys3 = read3D "keys3"
    let values3 = read3D "values3"

    [<TestMethod>]
    member this.Test1 () =
        let (nextToken1, keyCache1, valueCache1) = generateNextToken model53M Array.empty Array.empty 0 1 mostLikely
        Assert.AreEqual<int>(9038, nextToken1)
        compare3D "keyCache" keys1 keyCache1
        compare3D "valueCache" values1 valueCache1

    [<TestMethod>]
    member this.Test2 () =
        let (nextToken2, keyCache2, valueCache2) = generateNextToken model53M [| keys1|] [|values1|] 1 9038 mostLikely
        Assert.AreEqual<int>(2501, nextToken2)
        compare3D "keyCache" keys2 keyCache2
        compare3D "valueCache" values2 valueCache2

    [<TestMethod>]
    member this.Test3 () =
        let (nextToken3, keyCache3, valueCache3) = generateNextToken model53M [| keys1; keys2|] [|values1; values2 |] 2 2501 mostLikely
        Assert.AreEqual<int>(263, nextToken3)
        compare3D "keyCache" keys3 keyCache3
        compare3D "valueCache" values3 valueCache3