namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open HelperFunctions

[<TestClass>]
type ReshapeToMultipleHeadsTests() =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input = [| 1.0; 2.0; 3.0; 4.0 |]
        let headSize = 2
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
        let headSize = 3
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
        let headSize = 4
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0 |]; [| 5.0; 6.0; 7.0; 8.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
        let headSize = 6
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]; [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0 |]
        let headSize = 5
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0; 5.0 |]; [| 6.0; 7.0; 8.0; 9.0; 10.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
        let headSize = 4
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0 |]; [| 5.0; 6.0; 7.0; 8.0 |]; [| 9.0; 10.0; 11.0; 12.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0; 13.0; 14.0; 15.0; 16.0 |]
        let headSize = 8
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]; [| 9.0; 10.0; 11.0; 12.0; 13.0; 14.0; 15.0; 16.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0; 20.0 |]
        let headSize = 5
        let actual = reshapeToMultipleHeads headSize input
        let expected = [| [| 1.0; 2.0; 3.0; 4.0; 5.0 |]; [| 6.0; 7.0; 8.0; 9.0; 10.0 |]; [| 11.0; 12.0; 13.0; 14.0; 15.0 |]; [| 16.0; 17.0; 18.0; 19.0; 20.0 |] |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual<int>(expected.[i].Length, actual.[i].Length)
            for j in 0 .. expected.[i].Length - 1 do
                Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance)