namespace HelperFunctions

open HelperFunctions
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type SoftmaxTests () =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input = [| 1.0 |]
        let actual = softMax input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(1.0, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| -3.14 |]
        let actual = softMax input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(1.0, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| 1.0; 0.5 |]
        let actual = softMax input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(0.6224593312018546, actual.[0], tolerance)
        Assert.AreEqual(0.37754066879814546, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| 10.0; 0; -10.0  |]
        let actual = softMax input
        printfn "%A" actual
        Assert.AreEqual<int>(3, actual.Length)
        Assert.AreEqual(0.999954600070331, actual.[0], tolerance)
        Assert.AreEqual(4.539786860886666E-05, actual.[1], tolerance)
        Assert.AreEqual(2.061060046209062E-09, actual.[2], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [|0.9051561145; 0.3754769234; 0.6549278978; 0.6613300931; 0.05779244518; 0.1082396318; 0.7285146597; 0.6601591611; 0.2577876229; 0.6000075207|]
        let actual = softMax input

        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(10, actual.Length)
        Assert.AreEqual(0.14468590985919486, actual.[0], tolerance)
        Assert.AreEqual(0.08519017096724825, actual.[1], tolerance)
        Assert.AreEqual(0.11265578703184624, actual.[2], tolerance)
        Assert.AreEqual(0.11337934509066327, actual.[3], tolerance)
        Assert.AreEqual(0.0620041665202172, actual.[4], tolerance)
        Assert.AreEqual(0.06521234368566953, actual.[5], tolerance)
        Assert.AreEqual(0.12125839920911094, actual.[6], tolerance)
        Assert.AreEqual(0.11324666328320446, actual.[7], tolerance)
        Assert.AreEqual(0.07573169480489797, actual.[8], tolerance)
        Assert.AreEqual(0.10663551954794727, actual.[9], tolerance)