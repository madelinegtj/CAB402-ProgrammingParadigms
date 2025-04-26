namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open HelperFunctions

[<TestClass>]
type FlattenMultipleHeadsTests() =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| [| 1.0 |]; [| 2.0 |]; [| 3.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| [| 1.0; 2.0; 3.0; 4.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let input = [| [| 1.0 |]; [| 2.0 |]; [| 3.0 |]; [| 4.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let input = [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |]; [| 7.0; 8.0; 9.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let input = [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |]; [| 5.0; 6.0 |]; [| 7.0; 8.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test9 () =
        let input = [| [| 1.0; 2.0; 3.0; 4.0 |]; [| 5.0; 6.0; 7.0; 8.0 |] |]
        let actual = flattenMultipleHeads input
        let expected = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test10 () =
        let input = [||]
        let actual = flattenMultipleHeads input
        let expected = [||]
        Assert.AreEqual<int>(expected.Length, actual.Length)