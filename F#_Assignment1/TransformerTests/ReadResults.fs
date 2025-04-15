module ReadResults

open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

let tolerance = 1E-10

let resultsLocation test = 
    System.IO.Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "../../../ExpectedResults", test + ".results")




let readArray1D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> reader.ReadDouble())

let readArray2D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> readArray1D reader)

let readArray3D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> readArray2D reader)




let read1D test =
    let fileName = resultsLocation test
    use reader = new BinaryReader(File.OpenRead(fileName))
    readArray1D reader

let read2D test =
    let fileName = resultsLocation test
    use reader = new BinaryReader(File.OpenRead(fileName))
    readArray2D reader

let read3D test =
    let fileName = resultsLocation test
    use reader = new BinaryReader(File.OpenRead(fileName))
    readArray3D reader




let compare1D name (expected: float[]) (actual: float[]) : unit =
    Assert.AreEqual<int>(Array.length expected, Array.length actual)
    for i in [0 .. expected.Length-1] do
        Assert.AreEqual(expected.[i], actual.[i], tolerance, $"{name}.[{i}]")

let compare2D name (expected: float[][]) (actual: float[][]) : unit =
    Assert.AreEqual<int>(expected.Length, actual.Length)
    for i in [0 .. expected.Length-1] do
        for j in [0 .. expected.[i].Length-1] do
            Assert.AreEqual(expected.[i].[j], actual.[i].[j], tolerance, $"{name}.[{i}].[{j}]")

let compare3D name (expected: float[][][]) (actual: float[][][]) : unit =
    Assert.AreEqual<int>(expected.Length, actual.Length)
    for i in [0 .. expected.Length-1] do
        for j in [0 .. expected.[i].Length-1] do
            for k in [0 .. expected.[j].Length-1] do
            Assert.AreEqual(expected.[i].[j].[k], actual.[i].[j].[k], tolerance, $"{name}.[{i}].[{j}].[{k}]")