namespace TransformerFunctions

open Transformer
open Microsoft.VisualStudio.TestTools.UnitTesting
open ReadResults
open TransformerTestsHelpers

[<TestClass>]
type FeedForwardOneLayerTests () =

    let model53M = readModel53M

    let keys1 = read3D "keys1"
    let values1 = read3D "values1"
    let keys2 = read3D "keys2"
    let values2 = read3D "values2"
    let keys3 = read3D "keys3"
    let values3 = read3D "values3"
    let output1 = read2D "output1"
    let output2 = read2D "output2"
    let output3 = read2D "output3"

    let tolerance = 0

    [<TestMethod>]
    member this.Test10 () =
        let (output10, keyCache10, valueCache10) = feedforwardOneLayer model53M Array.empty Array.empty 0  model53M.tokenEmbedding[1] 0
        compare1D "output10" output1.[0] output10
        compare2D "keyCache10" keys1.[0] keyCache10
        compare2D "valueCache10" values1.[0] valueCache10   

    [<TestMethod>]
    member this.Test11 () =
        let (output11, keyCache11, valueCache11) = feedforwardOneLayer model53M Array.empty Array.empty 0  output1.[0] 1
        compare1D "output11" output1.[1] output11
        compare2D "keyCache11" keys1.[1] keyCache11
        compare2D "valueCache11" values1.[1] valueCache11   

    [<TestMethod>]
    member this.Test12 () =
        let (output12, keyCache12, valueCache12) = feedforwardOneLayer model53M Array.empty Array.empty 0 output1.[1] 2
        compare1D "output12" output1.[2] output12
        compare2D "keyCache12" keys1.[2] keyCache12
        compare2D "valueCache12" values1.[2] valueCache12           
     
    [<TestMethod>]
    member this.Test13 () =     
        let (output13, keyCache13, valueCache13) = feedforwardOneLayer model53M Array.empty Array.empty 0 output1.[2] 3
        compare1D "output13" output1.[3] output13
        compare2D "keyCache13" keys1.[3] keyCache13
        compare2D "valueCache13" values1.[3] valueCache13    
        
    [<TestMethod>]
    member this.Test14 () =
        let (output14, keyCache14, valueCache14) = feedforwardOneLayer model53M Array.empty Array.empty 0 output1.[3] 4
        compare1D "output14" output1.[4] output14
        compare2D "keyCache14" keys1.[4] keyCache14
        compare2D "valueCache14" values1.[4] valueCache14   
        
    [<TestMethod>]
    member this.Test15 () =
        let (output15, keyCache15, valueCache15) = feedforwardOneLayer model53M Array.empty Array.empty 0 output1.[4] 5
        compare1D "output15" output1.[5] output15
        compare2D "keyCache15" keys1.[5] keyCache15
        compare2D "valueCache15" values1.[5] valueCache15   

    [<TestMethod>]
    member this.Test20 () =
        let (output20, keyCache20, valueCache20) = feedforwardOneLayer model53M [|keys1|] [|values1|]  1  model53M.tokenEmbedding[9038] 0
        compare1D "output20" output2.[0] output20
        compare2D "keyCache20" keys2.[0] keyCache20
        compare2D "valueCache20" values2.[0] valueCache20   

    [<TestMethod>]
    member this.Test21 () =
        let (output21, keyCache21, valueCache21) = feedforwardOneLayer model53M [|keys1|] [|values1|]  1  output2.[0] 1
        compare1D "output21" output2.[1] output21
        compare2D "keyCache21" keys2.[1] keyCache21
        compare2D "valueCache21" values2.[1] valueCache21   

    [<TestMethod>]
    member this.Test22 () =
        let (output22, keyCache22, valueCache22) = feedforwardOneLayer model53M [|keys1|] [|values1|] 1 output2.[1] 2
        compare1D "output22" output2.[2] output22
        compare2D "keyCache22" keys2.[2] keyCache22
        compare2D "valueCache22" values2.[2] valueCache22           
     
    [<TestMethod>]
    member this.Test23 () =     
        let (output23, keyCache23, valueCache23) = feedforwardOneLayer model53M [|keys1|] [|values1|] 1 output2.[2] 3
        compare1D "output23" output2.[3] output23
        compare2D "keyCache23" keys2.[3] keyCache23
        compare2D "valueCache23" values2.[3] valueCache23    
        
    [<TestMethod>]
    member this.Test24 () =
        let (output24, keyCache24, valueCache24) = feedforwardOneLayer model53M [|keys1|] [|values1|] 1 output2.[3] 4
        compare1D "output24" output2.[4] output24
        compare2D "keyCache24" keys2.[4] keyCache24
        compare2D "valueCache24" values2.[4] valueCache24   
        
    [<TestMethod>]
    member this.Test25 () =
        let (output25, keyCache25, valueCache25) = feedforwardOneLayer model53M [|keys1|] [|values1|] 1 output2.[4] 5
        compare1D "output25" output2.[5] output25
        compare2D "keyCache25" keys2.[5] keyCache25
        compare2D "valueCache25" values2.[5] valueCache25   


    [<TestMethod>]
    member this.Test30 () =
        let (output30, keyCache30, valueCache30) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2  model53M.tokenEmbedding[2501] 0
        compare1D "output30" output3.[0] output30
        compare2D "keyCache30" keys3.[0] keyCache30
        compare2D "valueCache30" values3.[0] valueCache30   

    [<TestMethod>]
    member this.Test31 () =
        let (output31, keyCache31, valueCache31) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2  output3.[0] 1
        compare1D "output31" output3.[1] output31
        compare2D "keyCache31" keys3.[1] keyCache31
        compare2D "valueCache31" values3.[1] valueCache31   

    [<TestMethod>]
    member this.Test32 () =
        let (output32, keyCache32, valueCache32) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2 output3.[1] 2
        compare1D "output32" output3.[2] output32
        compare2D "keyCache32" keys3.[2] keyCache32
        compare2D "valueCache32" values3.[2] valueCache32           
     
    [<TestMethod>]
    member this.Test33 () =     
        let (output33, keyCache33, valueCache33) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2 output3.[2] 3
        compare1D "output33" output3.[3] output33
        compare2D "keyCache33" keys3.[3] keyCache33
        compare2D "valueCache33" values3.[3] valueCache33    
        
    [<TestMethod>]
    member this.Test34 () =
        let (output34, keyCache34, valueCache34) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2 output3.[3] 4
        compare1D "output34" output3.[4] output34
        compare2D "keyCache34" keys3.[4] keyCache34
        compare2D "valueCache34" values3.[4] valueCache34   
        
    [<TestMethod>]
    member this.Test35 () =
        let (output35, keyCache35, valueCache35) = feedforwardOneLayer model53M [|keys1;keys2|] [|values1;values2|]  2 output3.[4] 5
        compare1D "output35" output3.[5] output35
        compare2D "keyCache35" keys3.[5] keyCache35
        compare2D "valueCache35" values3.[5] valueCache35   