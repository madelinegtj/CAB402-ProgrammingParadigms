namespace HelperFunctions

open Types
open HelperFunctions
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type ElementWiseMultiplyTests() =

    let tolerance = 0.0

    [<TestMethod>]
    member this.Test1 () =
        let a = [| 1.0; 2.0; 3.0 |]
        let b = [| 4.0; 5.0; 6.0 |]
        let actual = elementWiseMultiply a b
        let expected = [| 4.0; 10.0; 18.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let a = [| 0.0; 0.0; 0.0 |]
        let b = [| 1.0; 2.0; 3.0 |]
        let actual = elementWiseMultiply a b
        let expected = [| 0.0; 0.0; 0.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let a = [| -1.0; -2.0; -3.0 |]
        let b = [| 4.0; 5.0; 6.0 |]
        let actual = elementWiseMultiply a b
        let expected = [| -4.0; -10.0; -18.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let a = [| -1.0; 2.0; -3.0 |]
        let b = [| 4.0; -5.0; 6.0 |]
        let actual = elementWiseMultiply a b
        let expected = [| -4.0; -10.0; -18.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let a = [| 2.0 |]
        let b = [| 3.0 |]
        let actual = elementWiseMultiply a b
        let expected = [| 6.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        Assert.AreEqual(expected.[0], actual.[0], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let a = [||]
        let b = [||]
        let actual = elementWiseMultiply a b
        let expected = [||]
        Assert.AreEqual<int>(expected.Length, actual.Length)

    [<TestMethod>]
    member this.Test7 () =
        let a = [| 1e10; 2e10; 3e10 |]
        let b = [| 4e10; 5e10; 6e10 |]
        let actual = elementWiseMultiply a b
        let expected = [| 4e20; 1e21; 1.8e21 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let a = [| 1e-10; 2e-10; 3e-10 |]
        let b = [| 4e-10; 5e-10; 6e-10 |]
        let actual = elementWiseMultiply a b
        actual |> Array.iter (fun f -> printfn "%s" (f.ToString("R")))
        let expected = [| 4.0000000000000004E-20; 1.0000000000000001E-19; 1.8E-19 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)