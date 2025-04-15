namespace HelperFunctions

open HelperFunctions
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type RootMeanSquareNormalizeTests () =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input = [| 3.14 |]
        let weights = [| 1.0 |]
        let actual = rootMeanSquareNormalize weights input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(0.9999994928804214, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| 3.14 |]
        let weights = [| -5.3 |]
        let actual = rootMeanSquareNormalize weights input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(-5.299997312266234, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| 1.0; 1.0 |]
        let actual = rootMeanSquareNormalize weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(1.264900944900269, actual.[0], tolerance)
        Assert.AreEqual(0.6324504724501345, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| 2.5; -0.7 |]
        let actual = rootMeanSquareNormalize weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(3.1622523622506726, actual.[0], tolerance)
        Assert.AreEqual(-0.4427153307150941, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [| 10.0; 0; -10.0  |]
        let weights = [| 2.5; 0.5; -1.9 |]
        let actual = rootMeanSquareNormalize weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(3, actual.Length)
        Assert.AreEqual(3.061861948839335, actual.[0], tolerance)
        Assert.AreEqual(0, actual.[1], tolerance)
        Assert.AreEqual(2.3270150811178945, actual.[2], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let input = [|0.9051561145; 0.3754769234; 0.6549278978; 0.6613300931; 0.05779244518; 0.1082396318; 0.7285146597; 0.6601591611; 0.2577876229; 0.6000075207|]
        let weights = [|1.61482813; 1.220419508; -1.596362596; 1.912205073; 1.876581213; 1.881535854; 0.5799674148; -1.043039349; 0.9546514274; -1.54770726|]   
        let actual = rootMeanSquareNormalize weights input
        Assert.AreEqual<int>(10, actual.Length)
        Assert.AreEqual(2.5699815296379964, actual.[0], tolerance)
        Assert.AreEqual(0.8056985799463401, actual.[1], tolerance)
        Assert.AreEqual(-1.8382528170563517, actual.[2], tolerance)
        Assert.AreEqual(2.2234786195019174, actual.[3], tolerance)
        Assert.AreEqual(0.19068592607639512, actual.[4], tolerance)
        Assert.AreEqual(0.35807913127589397, actual.[5], tolerance)
        Assert.AreEqual(0.7428858656936139, actual.[6], tolerance)
        Assert.AreEqual(-1.2106805168856727, actual.[7], tolerance)
        Assert.AreEqual(0.43270019860613135, actual.[8], tolerance)
        Assert.AreEqual(-1.6327726620287253, actual.[9], tolerance)


        //actual |> Array.iteri (fun i x -> printfn "Assert.AreEqual(%s, actual.[%d], tolerance)" (x.ToString("R")) i)
        //Array.init 10 (fun _ -> System.Random().NextDouble() * 4.0 - 2.0)
        //printfn "%A" weights