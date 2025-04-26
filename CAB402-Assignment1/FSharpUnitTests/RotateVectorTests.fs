namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open System.Numerics
open HelperFunctions

[<TestClass>]
type RotateVectorTests() =

    let tolerance = 0.0

    [<TestMethod>]
    member this.Test1 () =
        let rotationCoefficients = [| Complex(1.0, 0.0) |]
        let input = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test2 () =
        let rotationCoefficients = [| Complex(2.0, 0.0); Complex(0.0, 2.0) |]
        let input = [| [| 1.0; 1.0; 2.0; 2.0 |]; [| 3.0; 3.0; 4.0; 4.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 2.0; 2.0; -4.0; 4.0 |]; [| 6.0; 6.0; -8.0; 8.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test3 () =
        let rotationCoefficients = [| Complex(1.0, 1.0); Complex(1.0, 1.0) |]
        let input = [| [| 1.0; 0.0; 0.0; 1.0 |]; [| 1.0; 1.0; 2.0; 2.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 1.0; 1.0; -1.0; 1.0 |]; [| 0.0; 2.0; 0.0; 4.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test4 () =
        let rotationCoefficients = [| Complex(0.0, 1.0); Complex(1.0, 0.0) |]
        let input = [| [| 1.0; 1.0; 1.0; 1.0 |]; [| 2.0; 2.0; 2.0; 2.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| -1.0; 1.0; 1.0; 1.0 |]; [| -2.0; 2.0; 2.0; 2.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test5 () =
        let rotationCoefficients = [| Complex(1.0, 0.0); Complex(1.0, 0.0) |]
        let input = [| [| 0.0; 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0; 7.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 0.0; 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0; 7.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test6 () =
        let rotationCoefficients = [| Complex(1.0, 1.0); Complex(0.0, 1.0) |]
        let input = [| [| 1.0; 1.0; 1.0; 1.0 |]; [| 2.0; 2.0; 2.0; 2.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 0.0; 2.0; -1.0; 1.0 |]; [| 0.0; 4.0; -2.0; 2.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test7 () =
        let rotationCoefficients = [| Complex(2.0, 2.0); Complex(2.0, 2.0) |]
        let input = [| [| 1.0; 1.0; 1.0; 1.0 |]; [| 2.0; 2.0; 2.0; 2.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 0.0; 4.0; 0.0; 4.0 |]; [| 0.0; 8.0; 0.0; 8.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")

    [<TestMethod>]
    member this.Test8 () =
        let rotationCoefficients = [| Complex(1.0, 0.0); Complex(0.0, 1.0) |]
        let input = [| [| 0.0; 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0; 7.0 |] |]
        let actual = rotateVector rotationCoefficients input
        printfn "%A" actual
        let expected = [| [| 0.0; 1.0; -3.0; 2.0 |]; [| 4.0; 5.0; -7.0; 6.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"element[{i}][{j}]")