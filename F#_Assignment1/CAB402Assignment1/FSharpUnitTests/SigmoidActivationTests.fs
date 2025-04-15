namespace HelperFunctions

open Microsoft.VisualStudio.TestTools.UnitTesting
open HelperFunctions
open Transformer

[<TestClass>]
type SigmoidActivationTests() =

    let tolerance = 0

    [<TestMethod>]
    member this.Test1 () =
        let input = [| 0.0; 1.0; -1.0 |]
        let actual = sigmoidActivation input
        let expected = [| 0.0; 0.7310585786300049; -0.2689414213699951 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let input = [| 2.0; -2.0; 0.5 |]
        let actual = sigmoidActivation input
        let expected = [| 1.7615941559557646; -0.2384058440442351; 0.3112296656009273 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let input = [| 10.0; -10.0; 0.1 |]
        let actual = sigmoidActivation input
        let expected = [| 9.999546021312977; -0.00045397868702434395; 0.052497918747894 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let input = [| 0.0; 0.0; 0.0 |]
        let actual = sigmoidActivation input
        let expected = [| 0.0; 0.0; 0.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let input = [| 1e-10; -1e-10; 1e-5 |]
        let actual = sigmoidActivation input
        actual |> Array.iter (fun f -> printfn "%s" (f.ToString("R")))
        let expected = [| 5.00000000025E-11; -4.9999999997500004E-11; 5.000025E-06 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let input = [| 1e10; -1e10 |]
        let actual = sigmoidActivation input
        actual |> Array.iter (fun f -> printfn "%s" (f.ToString("R")))
        let expected = [| 1e10; 0 |] // Sigmoid function saturates at large values
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let input = [| 0.5; -0.5; 0.25 |]
        let actual = sigmoidActivation input
        actual |> Array.iter (fun f -> printfn "%s" (f.ToString("R")))
        let expected = [| 0.3112296656009273; -0.1887703343990727; 0.14054412522144952 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let input = [| 0.0 |]
        let actual = sigmoidActivation input
        let expected = [| 0.0 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        Assert.AreEqual(expected.[0], actual.[0], tolerance)

    [<TestMethod>]
    member this.Test9 () =
        let input = [||]
        let actual = sigmoidActivation input
        let expected = [||]
        Assert.AreEqual<int>(expected.Length, actual.Length)