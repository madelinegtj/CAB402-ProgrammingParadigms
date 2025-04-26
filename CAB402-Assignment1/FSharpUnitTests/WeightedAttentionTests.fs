namespace AttentionFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open Attention

[<TestClass>]
type WeightedAttentionTests() =

    let tolerance = 0.0

    [<TestMethod>]
    member this.Test1 () =
        let query = [| 1.1; 2.2; 3.3 |]
        let valueLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 16.939999999999998
        Assert.AreEqual(expected, actual, tolerance, "Test1 failed.")

    [<TestMethod>]
    member this.Test2 () =
        let query = [| 0.0; 0.0; 0.0 |]
        let valueLookup = function
            | _ -> 1.1
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 0.0
        Assert.AreEqual(expected, actual, tolerance, "Test2 failed.")

    [<TestMethod>]
    member this.Test3 () =
        let query = [| 1.1; 1.1; 1.1; 1.1 |]
        let valueLookup = function
            | 0 -> 1.1
            | 1 -> 1.1
            | 2 -> 1.1
            | 3 -> 1.1
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 4.840000000000001
        Assert.AreEqual(expected, actual, tolerance, "Test3 failed.")

    [<TestMethod>]
    member this.Test4 () =
        let query = [| 1.1; 2.2; 3.3; 4.4 |]
        let valueLookup = function
            | 0 -> 4.4
            | 1 -> 3.3
            | 2 -> 2.2
            | 3 -> 1.1
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 24.2
        Assert.AreEqual(expected, actual, tolerance, "Test4 failed.")

    [<TestMethod>]
    member this.Test5 () =
        let query = [| 2.2; 2.2; 2.2; 2.2 |]
        let valueLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | 3 -> 4.4
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 24.200000000000003
        Assert.AreEqual(expected, actual, tolerance, "Test5 failed.")

    [<TestMethod>]
    member this.Test6 () =
        let query = [| 1.1; 2.2 |]
        let valueLookup = function
            | 0 -> 2.2
            | 1 -> 3.3
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 9.68
        Assert.AreEqual(expected, actual, tolerance, "Test6 failed.")

    [<TestMethod>]
    member this.Test7 () =
        let query = [| 3.3; 3.3; 3.3 |]
        let valueLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | _ -> 0.0
        let actual = attentionScore query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 12.574688862950051
        Assert.AreEqual(expected, actual, tolerance, "Test7 failed.")

    [<TestMethod>]
    member this.Test8 () =
        let query = [| 1.1; 2.2; 3.3; 4.4; 5.5; 6.6 |]
        let valueLookup = function
            | 0 -> 6.6
            | 1 -> 5.5
            | 2 -> 4.4
            | 3 -> 3.3
            | 4 -> 2.2
            | 5 -> 1.1
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 67.75999999999999
        Assert.AreEqual(expected, actual, tolerance, "Test8 failed.")

    [<TestMethod>]
    member this.Test9 () =
        let query = [| 1.1; 1.1; 1.1; 1.1; 1.1; 1.1 |]
        let valueLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | 3 -> 4.4
            | 4 -> 5.5
            | 5 -> 6.6
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 25.410000000000004
        Assert.AreEqual(expected, actual, tolerance, "Test9 failed.")

    [<TestMethod>]
    member this.Test10 () =
        let query = [| 2.2; 4.4; 6.6; 8.8; 10.1 |]
        let valueLookup = function
            | 0 -> 5.5
            | 1 -> 4.4
            | 2 -> 3.3
            | 3 -> 2.2
            | 4 -> 1.1
            | _ -> 0.0
        let actual = weightedAttention query valueLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 83.71000000000001
        Assert.AreEqual(expected, actual, tolerance, "Test10 failed.")