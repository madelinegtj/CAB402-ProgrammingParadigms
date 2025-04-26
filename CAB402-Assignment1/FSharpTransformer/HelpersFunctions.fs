module HelperFunctions

open System.Diagnostics
open System.Numerics
open Types

// Vector operations ...

// Computes the product of x and y by multiplying the matrix w by the vector x.
// Given matrix w with size m x n, the input vector should be of length m, and
// the output vector size n.
let matrixMultiply (weights: WeightMatrix) (input: Vector) : Vector =

    // For each row in the weight matrix
    Array.map (fun row -> 
        // Calculate the dot product of the current row and the input vector, then accumulates/sum them into acc
        Array.fold2 (fun acc w x -> acc + w * x) 0.0 row input
    ) weights

// Adds two vectors together element-wise, returns a new vector.
// Both vectors should be of the dimension (array size).
let add (a: Vector) (b: Vector) : Vector =

    // For each corresponding pair of elements x and y from vectors a and b respectively,
    // add them together to form the new vector.
    Array.map2 (fun x y -> x + y) a b

// Multiplies two vectors together element-wise, returns a new vector.
// Both vectors should be of the dimension (array size).
let elementWiseMultiply (a : Vector) (b: Vector) : Vector =  
    
    // For each corresponding pair of elements x and y from vectors a and b respectively,
    // multiply them to form the new vector.
    Array.map2 (fun x y -> x * y) a b

// Performs root mean square (RMS) layer normalization on an input vector.
// To apply RMS layer normalization, we compute:
//   rms = sqrt(mean(input ^ 2))
//   output[i] = (output[i] / rms) * weights[i]
// Weights is a learnt parameter that is computed during training.
let rootMeanSquareNormalize (weights: WeightVector) (input: Vector) : Vector =
    
    // Compute mean square of the input elements
    let meanSquare = (Array.sumBy (fun x -> x * x) input) / float input.Length + 1e-5
    // Calculate scaling factor by taking the inverse of root meanSquare
    let scale = 1.0 / sqrt meanSquare
    // For each (weight, input) pair, apply normalization and scaling
    Array.map2 (fun w x -> w * (scale * x)) weights input

// Applies the softmax function to the given input vector (array).
// Softmax is a function that transforms a vector into a probability distribution,
// ranging from 0 to 1. Softmax is computed as:
//   softmax(xi) = exp(xi) / sum(exp(xj))
let softMax (input: Vector) : Vector =
    
    // Find the maximum value of input vector for numerical stability
    let max = Array.fold max System.Double.MinValue input
    // For each input element, calculate the exponentials of (input - max)
    let expValues = Array.map (fun x -> exp(x - max)) input
    let sum = Array.sum expValues   // Sum all the exponentials

    //Normalize each exponential by dividing by the sum
    Array.map (fun x -> x / sum) expValues

// Applies our activation function: Sigmoid Linear Unit (SilU)
// SilU is computed as silu(x) = x*σ(x).
// σ(x) is the logistic sigmoid, or σ(x) = 1 / 1 + exp(-x). 
let sigmoidActivation (input:Vector) : Vector =

    let sigmoid x = 1.0 / (1.0 + exp(-x))
    // Apply SiLU(x) to each element in the input
    Array.map (fun x -> x * sigmoid(x)) input


// Reshaping function ...

// To operate on each head of multi-head attention independently, each head
// needs its own sub-array. This function is responsible for dividing up the
// input vector into head-sized vectors, which are also arrays.
let reshapeToMultipleHeads (headSize: int)(input: Vector) : MultiHead = 
    Debug.Assert(input.Length % headSize = 0)
    input |> Array.chunkBySize headSize

// After computing multi-head attention, this function is responsible for
// recombining the separate head vectors into a single vector that forms
// the output for the layer.
let flattenMultipleHeads (input:MultiHead) : Vector =
    input |> Array.concat

// Converts the vector for each head into a series of two-dimensional positions
// suitable for Rotary Position Embedding (RoPE). The two dimensions are represented
// using a complex number type, with real and imaginary parts corresponding to the
// X and Y coordinates respectively.
let toComplex (vector: Vector) : Complex[] =
    Debug.Assert(vector.Length % 2 = 0)
    vector
    |> Array.chunkBySize 2
    |> Array.map (fun [|real;imag|] -> Complex(real,imag))

// Converts the list of 2D coordinates back to a single vector after applying
// Rotary Postion Embedding.
let flattenComplex (vector: Complex[]): Vector =    
    vector |> Array.collect (fun c -> [|c.Real; c.Imaginary|])



// Rotation functions ...

// Applies Rotary Position Embedding to a single pair of coordinates.
// i.e. for each pair of 2D coordinates, multiply by the corresponding rotationCoefficients.
let rotateOneHead (rotationCoeffients: Complex[]) (input: Complex[]) : Complex[] =
    
    // For each corresponding pair (coefficient, value) from rotationCoefficients and input array,
    // multiply the two complex numbers to rotate the input.
    Array.map2 (fun coef value -> coef * value) rotationCoeffients input

// Applies Rotary Position Embedding to each head of the input vector.
// You should use the utility functions above to convert the input vector into a series
// of 2D points, rotate them, and then merge it back to a single vector for each head.
let rotateVector (rotationCoeffients: Complex[]) (input: MultiHead) : MultiHead = 
    
    input
    |> Array.map (fun head ->       // For each head,
        // Convert the head (real-value vectors) into complex numbers
        let complexHead = toComplex head
        // Apply rotary position embedding; rotate the complex numbers using rotationCoefficients
        rotateOneHead rotationCoeffients complexHead
        // Flatten the rotated complex numbers back to real-value vector
        |> flattenComplex
    )
