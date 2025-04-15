namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open HelperFunctions
open System.Numerics

[<TestClass>]
type ToComplexTests() =

    let tolerance = 1e-10

    [<TestMethod>]
    member this.Test1 () =
        let input = [| 1.0; 2.0; 3.0; 4.0 |]
        let actual = toComplex input
        let expected = [| Complex(1.0, 2.0); Complex(3.0, 4.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| 0.0; 0.0; 0.0; 0.0 |]
        let actual = toComplex input
        let expected = [| Complex(0.0, 0.0); Complex(0.0, 0.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| -1.0; -2.0; -3.0; -4.0 |]
        let actual = toComplex input
        let expected = [| Complex(-1.0, -2.0); Complex(-3.0, -4.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| 1.0; -1.0; -1.0; 1.0 |]
        let actual = toComplex input
        let expected = [| Complex(1.0, -1.0); Complex(-1.0, 1.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [| 1e-10; 1e-10; -1e-10; -1e-10 |]
        let actual = toComplex input
        let expected = [| Complex(1e-10, 1e-10); Complex(-1e-10, -1e-10) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let input = [| 1e10; 1e10; -1e10; -1e10 |]
        let actual = toComplex input
        let expected = [| Complex(1e10, 1e10); Complex(-1e10, -1e10) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let input = [| 1.0; 2.0 |]
        let actual = toComplex input
        let expected = [| Complex(1.0, 2.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        Assert.AreEqual(expected.[0].Real, actual.[0].Real, tolerance)
        Assert.AreEqual(expected.[0].Imaginary, actual.[0].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let input = [| 0.0; 0.0 |]
        let actual = toComplex input
        let expected = [| Complex(0.0, 0.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        Assert.AreEqual(expected.[0].Real, actual.[0].Real, tolerance)
        Assert.AreEqual(expected.[0].Imaginary, actual.[0].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test9 () =
        let input = [| -1.0; -2.0 |]
        let actual = toComplex input
        let expected = [| Complex(-1.0, -2.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        Assert.AreEqual(expected.[0].Real, actual.[0].Real, tolerance)
        Assert.AreEqual(expected.[0].Imaginary, actual.[0].Imaginary, tolerance)