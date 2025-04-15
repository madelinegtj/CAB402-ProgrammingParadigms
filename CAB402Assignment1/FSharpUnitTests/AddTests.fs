namespace HelperFunctions

open HelperFunctions
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type AddTests () =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input1 = [| 1.0 |]
        let input2 = [| 2.0 |] 
        let actual = add input1 input2
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(3.0, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input1 = [| 0.5; 0 |]
        let input2 = [| -0.1; 3.14 |] 
        let actual = add input1 input2
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(0.4, actual.[0], tolerance)
        Assert.AreEqual(3.14, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input1 = [| |]
        let input2 = [| |] 
        let actual = add input1 input2
        Assert.AreEqual<int>(0, actual.Length)

    [<TestMethod>]
    member this.Test4 () =
        let input1 = [|-1.466447311; 1.887976352; 1.129782154; 1.504959817; 0.3727634623; 1.781528633; -1.540429924; -1.388416314; 0.896099117|]
        let input2 = [|-0.4452849493; -0.8866154537; -0.2760880058; 1.77773588; -0.02603431151; -0.9116047027; 1.824632557; 0.1950688053; 1.194600764|]
        let actual = add input1 input2
        Assert.AreEqual<int>(9, actual.Length)
        Assert.AreEqual(-1.9117322603, actual.[0], tolerance)
        Assert.AreEqual(1.0013608983, actual.[1], tolerance)
        Assert.AreEqual(0.8536941482, actual.[2], tolerance)
        Assert.AreEqual(3.2826956970000003, actual.[3], tolerance)
        Assert.AreEqual(0.34672915079, actual.[4], tolerance)
        Assert.AreEqual(0.8699239302999999, actual.[5], tolerance)
        Assert.AreEqual(0.28420263300000004, actual.[6], tolerance)
        Assert.AreEqual(-1.1933475087, actual.[7], tolerance)
        Assert.AreEqual(2.090699881, actual.[8], tolerance)