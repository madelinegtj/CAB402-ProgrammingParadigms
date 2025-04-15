module SaveResults

open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

let saveResults2 test (output: float[]) (keyCache:float[][]) (valueCache:float[][]) : unit =
    let fileName = test + ".results"
    use writer = new BinaryWriter(File.OpenWrite(fileName))

    let writeArray1D (array:float[]) =
        writer.Write(array.Length)
        for v in array do
            writer.Write(v)

    let writeArray2D (array:float[][]) =
        writer.Write(array.Length)
        for v in array do
            writeArray1D v        

    writeArray1D output
    writeArray2D keyCache
    writeArray2D valueCache

let readArray1D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> reader.ReadDouble())

let readArray2D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> readArray1D reader)

let readArray3D (reader:BinaryReader) =
    let length = reader.ReadInt32()
    Array.init length (fun i -> readArray2D reader)

let readResults2D test =
    use reader = new BinaryReader(File.OpenRead(test + ".results"))
    let output = readArray1D reader
    let keyCache = readArray2D reader
    let valueCache = readArray2D reader
    (output, keyCache, valueCache)

let readResults3D test =
    use reader = new BinaryReader(File.OpenRead(test + ".results"))
    let output = readArray1D reader
    let keyCache = readArray3D reader
    let valueCache = readArray3D reader
    (output, keyCache, valueCache)

let saveResults3 test (output: float[]) (keyCache:float[][][]) (valueCache:float[][][]) : unit =
    let fileName = test + ".results"
    use writer = new BinaryWriter(File.OpenWrite(fileName))

    let writeArray1D (array:float[]) =
        writer.Write(array.Length)
        for v in array do
            writer.Write(v)

    let writeArray2D (array:float[][]) =
        writer.Write(array.Length)
        for v in array do
            writeArray1D v        

    let writeArray3D (array:float[][][]) =
        writer.Write(array.Length)
        for v in array do
            writeArray2D v   

    writeArray1D output
    writeArray3D keyCache
    writeArray3D valueCache

let tolerance = 1E-10

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

let compareResults2D test (output: float[]) (keyCache:float[][]) (valueCache:float[][]) : unit =
    let (expectedOutput, expectedKeys, expectedValues) = readResults2D test
    compare1D "output" expectedOutput output
    compare2D "keyCache" expectedKeys keyCache
    compare2D "valueCache" expectedValues valueCache

let compareResults3D test (output: float[]) (keyCache:float[][][]) (valueCache:float[][][]) : unit =
    let (expectedOutput, expectedKeys, expectedValues) = readResults3D test
    compare1D "output" expectedOutput output
    compare3D "keyCache" expectedKeys keyCache
    compare3D "valueCache" expectedValues valueCache

let compareCache test (keyCache:float[][][]) (valueCache:float[][][]) : unit =
    let (expectedOutput, expectedKeys, expectedValues) = readResults3D test
    compare3D "keyCache" expectedKeys keyCache
    compare3D "valueCache" expectedValues valueCache