module IO

open System.Numerics
open System.Text
open System.IO
open Types

/// Reads a single precision 32-bit floating point value from the input stream,
/// and converts it to 64-bit double precision.
let readSingle (reader: BinaryReader) =
    float(reader.ReadSingle())

/// Reads an array of single precision 32-bit floating point values from the given
/// file/reader, converted to double precision. The array is assumed to be of size
/// dim.
let read1DArray (reader: BinaryReader) dim =
    Array.init dim (fun _ -> readSingle reader)

/// Reads a two-dimensional array of single precision 32-bit floating point values
/// from the given file/reader, in row-major order, converted to double precision.
/// This two-dimensional array is usually a matrix. The matrix is assumed to be of
/// size dim1 x dim2.
let read2DArray (reader: BinaryReader) dim1 dim2 =
    Array.init dim1 (fun _ -> read1DArray reader dim2)

/// Reads a three-dimensional array of single precision 32-bit floating point values
/// from the given file/reader, in row-major order, converted to double precision.
/// These usually represent an array of matrices. Each matrix is assumed to be of
/// size dim2 x dim3, with dim1 matrices total read from the file.
let read3DArray (reader: BinaryReader) dim1 dim2 dim3 =
    Array.init dim1 (fun _ -> read2DArray reader dim2 dim3)

let readVocabFile filePath size =
    use reader = new BinaryReader(File.OpenRead(filePath))
     
    Array.init size (fun i ->
            let len = reader.ReadInt32()
            let bytes = reader.ReadBytes(len)
            let chars = Encoding.Default.GetChars(bytes)
            new string(chars))

// Read the entire model into memory. The file itself is not easily interpretable, but
// the structure mostly follows version 1 from llama2.c:
// https://github.com/karpathy/llama2.c/blob/master/export.py
let readModel modelFilePath vocabFilePath =
    use reader = new BinaryReader(File.OpenRead(modelFilePath))

    let dimension = reader.ReadInt32()
    let hiddenDimension = reader.ReadInt32();
    let numberOfLayers = reader.ReadInt32();
    let numberOfHeads = reader.ReadInt32();
    let numberOfKeyValueHeads = reader.ReadInt32();
    let vocabularySize = reader.ReadInt32();
    let seqenceLength = reader.ReadInt32();
    let headSize = dimension / numberOfHeads

    let tokenEmbedding = read2DArray reader vocabularySize dimension
    let normalizeInputWeights = read2DArray reader  numberOfLayers dimension
    let wq = read3DArray reader numberOfLayers dimension dimension
    let wk = read3DArray reader numberOfLayers dimension dimension
    let wv = read3DArray reader numberOfLayers dimension dimension
    let wo = read3DArray reader numberOfLayers dimension dimension
    let normalizeAttentionWeights = read2DArray reader numberOfLayers dimension
    let w1 = read3DArray reader numberOfLayers hiddenDimension dimension
    let w2 = read3DArray reader numberOfLayers dimension hiddenDimension
    let w3 = read3DArray reader numberOfLayers hiddenDimension dimension
    let normalizeOutputWeights = read1DArray reader dimension
    let rotationCoefficientsReal = read2DArray reader seqenceLength (headSize / 2)
    let rotationCoefficientsImaginary = read2DArray reader seqenceLength (headSize / 2)

    {
        headSize = headSize
        numberOfHeads = numberOfHeads
        numberOfLayers = numberOfLayers
        vocabularySize = vocabularySize
        tokenEmbedding = tokenEmbedding
        seqenceLength = seqenceLength
        normalizeOutputWeights = normalizeOutputWeights

        rotationCoefficients =     
            Array.init rotationCoefficientsReal.Length (fun i ->
                Array.init rotationCoefficientsReal.[i].Length (fun j ->
                    Complex(rotationCoefficientsReal.[i].[j], rotationCoefficientsImaginary.[i].[j])))

        weights = 
            Array.init numberOfLayers (fun layer ->
                {
                    normalizeInputWeights = normalizeInputWeights.[layer]
                    wq = wq.[layer]
                    wk = wk.[layer]
                    wv = wv.[layer]
                    wo = wo.[layer]
                    normalizeAttentionWeights = normalizeAttentionWeights.[layer]
                    w1 = w1.[layer]
                    w2 = w2.[layer]
                    w3 = w3 .[layer]      
                })

        vocabulary = readVocabFile vocabFilePath vocabularySize
    }

// Displays each token to the output console.
let printStory (story: string seq): unit =
    story
    |> Seq.iter (printf "%s")
    ; printfn "\n\n"