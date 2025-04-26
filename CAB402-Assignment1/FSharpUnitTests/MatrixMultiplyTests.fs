namespace HelperFunctions

open HelperFunctions
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type MatrixMultiplyTests () =

    let tolerance = 0

    [<TestMethod>]
    member this.Test01 () =
        let input = [| 3.14 |]
        let weights = [| [| 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(3.14, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test02 () =
        let input = [| 3.14 |]
        let weights = [| [| -2.5 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(-7.8500000000000005, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test03 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| [| 1.0; 1.0 |] |]
        let actual = matrixMultiply weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(1.5, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test04 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| [| 2.5; -0.7 |] |]
        let actual = matrixMultiply weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(2.15, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test05 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| [| 2.5; -0.7 |]; [| 0.65; -8.9 |] |]
        let actual = matrixMultiply weights input
        actual |> Array.iter (fun x -> printfn "%s" (x.ToString("R")))
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(2.15, actual.[0], tolerance)
        Assert.AreEqual(-3.8000000000000003, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test06 () =
        let input = [| 0.0; 0.0 |]
        let weights = [| [| 0.0; 0.0 |]; [| 0.0; 0.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(0.0, actual.[0], tolerance)
        Assert.AreEqual(0.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test07 () =
        let input = [| 1.0 |]
        let weights = [| [| 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(1, actual.Length)
        Assert.AreEqual(1.0, actual.[0], tolerance)

    [<TestMethod>]
    member this.Test08 () =
        let input = [| 1.0; 1.0 |]
        let weights = [| [| 1.0; 1.0 |]; [| 1.0; 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(2.0, actual.[0], tolerance)
        Assert.AreEqual(2.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test09 () =
        let input = [| -1.0; 1.0 |]
        let weights = [| [| 1.0; -1.0 |]; [| -1.0; 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(-2.0, actual.[0], tolerance)
        Assert.AreEqual(2.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test10 () =
        let input = [| 1.0; 0.5 |]
        let weights = [| [| 2.5; -0.7 |]; [| 0.65; -8.9 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(2.15, actual.[0], tolerance)
        Assert.AreEqual(-3.8000000000000003, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test11 () =
        let input = [| 1.0; 2.0; 3.0 |]
        let weights = [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(14.0, actual.[0], tolerance)
        Assert.AreEqual(32.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test12 () =
        let input = [| 1.0; -1.0; 1.0 |]
        let weights = [| [| 1.0; -1.0; 1.0 |]; [| -1.0; 1.0; -1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(3.0, actual.[0], tolerance)
        Assert.AreEqual(-3.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test13 () =
        let input = [| 0.5; 0.7; 0.8 |]
        let weights = [| [| 0.5; 0.3; 0.1 |]; [| -0.9; 1.5; 2.5 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(0.54, actual.[0], tolerance)
        Assert.AreEqual(2.5999999999999996, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test14 () =
        let input = [| 1.0; 2.0; 3.0; 4.0 |]
        let weights = [| [| 1.0; 2.0; 3.0; 4.0 |]; [| 4.0; 3.0; 2.0; 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(30.0, actual.[0], tolerance)
        Assert.AreEqual(20.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test15 () =
        let input = [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
        let weights = [| [| 1.0; 2.0; 3.0; 4.0; 5.0 |]; [| 5.0; 4.0; 3.0; 2.0; 1.0 |] |]
        let actual = matrixMultiply weights input
        Assert.AreEqual<int>(2, actual.Length)
        Assert.AreEqual(55.0, actual.[0], tolerance)
        Assert.AreEqual(35.0, actual.[1], tolerance)

    [<TestMethod>]
    member this.Test16 () =
        let input = [|0.3459181977; 1.72207769; 1.778685959; 0.3408925456; 0.9771234338; 0.8028557876|]
        let weights = [|[|0.7561527567; -0.1096155327; 1.342414097; 0.299215445; -0.3245152885; 0.795783616|];
              [|-0.4178604221; -1.077254168; 0.7893072418; -0.6271981282; 0.5389685594; 1.154277948|];
              [|1.797366931; -0.8614781738; 1.024130179; -1.965086605; 0.9627189739; -1.959609197|];
              [|0.4405074604; 0.4546365813; -0.9111027627; 1.037474876; 0.8037975438; -0.5078412973|]|]

        let actual = matrixMultiply weights input

        actual |> Array.iteri (fun i x -> printfn "%s %d" (x.ToString("R")) i)
        
        Assert.AreEqual<int>(4, actual.Length)
        Assert.AreEqual(2.8843419444330047, actual.[0], tolerance)
        Assert.AreEqual(0.6438191890557385, actual.[1], tolerance)
        Assert.AreEqual(-0.3426561353573354, actual.[2], tolerance)
        Assert.AreEqual(0.046086912102923594, actual.[3], tolerance)


        //let input = Array.init 6 (fun _ -> System.Random().NextDouble() * 4.0 - 2.0)
        //let weights = Array.init 4 (fun _ -> Array.init 6 (fun _ -> System.Random().NextDouble() * 4.0 - 2.0))
        //printfn "%A" input
        //printfn "%A" weights