namespace AttentionFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open Attention

[<TestClass>]
type AttentionScoreTests() =

    let tolerance = 0.0

    [<TestMethod>]
    member this.Test1 () =
        let query = [| 1.1; 2.2; 3.3 |]
        let keyLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 9.78031356007226
        Assert.AreEqual(expected, actual, tolerance, "Test1 failed.")

    [<TestMethod>]
    member this.Test2 () =
        let query = [| 0.0; 0.0; 0.0 |]
        let keyLookup = function
            | _ -> 1.1
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 0.0
        Assert.AreEqual(expected, actual, tolerance, "Test2 failed.")

    [<TestMethod>]
    member this.Test3 () =
        let query = [| 1.1; 1.1; 1.1; 1.1 |]
        let keyLookup = function
            | 0 -> 1.1
            | 1 -> 1.1
            | 2 -> 1.1
            | 3 -> 1.1
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 2.4200000000000004
        Assert.AreEqual(expected, actual, tolerance, "Test3 failed.")

    [<TestMethod>]
    member this.Test4 () =
        let query = [| 1.1; 2.2; 3.3; 4.4 |]
        let keyLookup = function
            | 0 -> 4.4
            | 1 -> 3.3
            | 2 -> 2.2
            | 3 -> 1.1
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 12.1
        Assert.AreEqual(expected, actual, tolerance, "Test4 failed.")

    [<TestMethod>]
    member this.Test5 () =
        let query = [| 2.2; 2.2; 2.2; 2.2 |]
        let keyLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | 3 -> 4.4
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 12.100000000000001
        Assert.AreEqual(expected, actual, tolerance, "Test5 failed.")

    [<TestMethod>]
    member this.Test6 () =
        let query = [| 1.1; 2.2 |]
        let keyLookup = function
            | 0 -> 2.2
            | 1 -> 3.3
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 6.844793641885779
        Assert.AreEqual(expected, actual, tolerance, "Test6 failed.")

    [<TestMethod>]
    member this.Test7 () =
        let query = [| 3.3; 3.3; 3.3 |]
        let keyLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 12.574688862950051
        Assert.AreEqual(expected, actual, tolerance, "Test7 failed.")

    [<TestMethod>]
    member this.Test8 () =
        let query = [| 1.1; 2.2; 3.3; 4.4; 5.5; 6.6 |]
        let keyLookup = function
            | 0 -> 6.6
            | 1 -> 5.5
            | 2 -> 4.4
            | 3 -> 3.3
            | 4 -> 2.2
            | 5 -> 1.1
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 27.662904161831356
        Assert.AreEqual(expected, actual, tolerance, "Test8 failed.")

    [<TestMethod>]
    member this.Test9 () =
        let query = [| 1.1; 1.1; 1.1; 1.1; 1.1; 1.1 |]
        let keyLookup = function
            | 0 -> 1.1
            | 1 -> 2.2
            | 2 -> 3.3
            | 3 -> 4.4
            | 4 -> 5.5
            | 5 -> 6.6
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 10.373589060686761
        Assert.AreEqual(expected, actual, tolerance, "Test9 failed.")

    [<TestMethod>]
    member this.Test10 () =
        let query = [| 2.2; 4.4; 6.6; 8.8; 10.1 |]
        let keyLookup = function
            | 0 -> 5.5
            | 1 -> 4.4
            | 2 -> 3.3
            | 3 -> 2.2
            | 4 -> 1.1
            | _ -> 0.0
        let actual = attentionScore query keyLookup
        printfn "%s" (actual.ToString("R"))
        let expected = 37.43625007930148
        Assert.AreEqual(expected, actual, tolerance, "Test10 failed.")