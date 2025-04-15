namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open System.Numerics
open HelperFunctions

[<TestClass>]
type RotateOneHeadTests() =

    let tolerance = 0.0

    [<TestMethod>]
    member this.Test1 () =
        let rotationCoefficients = [| Complex(1.0, 0.0); Complex(0.0, 1.0) |]
        let input = [| Complex(1.0, 2.0); Complex(3.0, 4.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(1.0, 2.0); Complex(-4.0, 3.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let rotationCoefficients = [| Complex(2.0, 0.0) |]
        let input = [| Complex(1.0, 1.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(2.0, 2.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let rotationCoefficients = [| Complex(1.0, 1.0); Complex(1.0, 1.0); Complex(1.0, 1.0) |]
        let input = [| Complex(1.0, 0.0); Complex(0.0, 1.0); Complex(1.0, 1.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(1.0, 1.0); Complex(-1.0, 1.0); Complex(0.0, 2.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let rotationCoefficients = [| Complex(0.0, 1.0); Complex(1.0, 0.0); Complex(1.0, 1.0) |]
        let input = [| Complex(1.0, 1.0); Complex(1.0, 1.0); Complex(1.0, 0.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(-1.0, 1.0); Complex(1.0, 1.0); Complex(1.0, 1.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let rotationCoefficients = [| Complex(1.0, 1.0); Complex(0.0, 1.0) |]
        let input = [| Complex(1.0, 1.0); Complex(1.0, 1.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(0.0, 2.0); Complex(-1.0, 1.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let rotationCoefficients = [| Complex(2.0, 2.0); Complex(2.0, 2.0); Complex(2.0, 2.0) |]
        let input = [| Complex(1.0, 1.0); Complex(1.0, 1.0); Complex(1.0, 1.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(0.0, 4.0); Complex(0.0, 4.0); Complex(0.0, 4.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let rotationCoefficients = [| Complex(1.0, 0.0) |]
        let input = [| Complex(0.0, 1.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(0.0, 1.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

    [<TestMethod>]
    member this.Test9() =
        let rotationCoefficients = [| Complex(2.0, 0.0); Complex(0.0, 2.0); Complex(1.0, 1.0) |]
        let input = [| Complex(1.0, 1.0); Complex(2.0, 2.0); Complex(1.0, 0.0) |]
        let actual = rotateOneHead rotationCoefficients input
        let expected = [| Complex(2.0, 2.0); Complex(-4.0, 4.0); Complex(1.0, 1.0) |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i].Real, actual.[i].Real, tolerance)
            Assert.AreEqual(expected.[i].Imaginary, actual.[i].Imaginary, tolerance)

