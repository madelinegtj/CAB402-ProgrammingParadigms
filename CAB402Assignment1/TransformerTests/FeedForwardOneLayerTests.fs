namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open SaveResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardOneLayerTests () =

    let model53M = readModel53M

    let input = model53M.tokenEmbedding[1]

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let (output, keyCache, valueCache) = feedforwardOneLayer model53M Array.empty Array.empty 0 input 0
        saveResults2 "test1a" output keyCache valueCache
        compareResults2D "test1a" output keyCache valueCache

    [<TestMethod>]
    member this.Test2 () =
        let (output0, keyCache0, valueCache0) = feedforwardOneLayer model53M Array.empty Array.empty 0 input 0
        let (output1, keyCache1, valueCache1) = feedforwardOneLayer model53M Array.empty Array.empty 0 output0 1
        saveResults2 "test2a" output1 keyCache1 valueCache1    
        compareResults2D "test2a" output1 keyCache1 valueCache1

    [<TestMethod>]
    member this.Test3 () =
        let (output0, keyCache0, valueCache0) = feedforwardOneLayer model53M Array.empty Array.empty 0 input 0
        let (output1, keyCache1, valueCache1) = feedforwardOneLayer model53M Array.empty Array.empty 0 output0 1
        let (output2, keyCache2, valueCache2) = feedforwardOneLayer model53M Array.empty Array.empty 0 output1 2
        saveResults2 "test3a" output2 keyCache2 valueCache2
        compareResults2D "test3a" output2 keyCache2 valueCache2