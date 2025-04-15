using System.Text;

public class Transformer
{
    int dim, hidden_dim, n_layers, n_heads, vocab_size, seq_len, head_size; /// Model constants.
    double[][] token_embedding; /// Learnt embedding vector for each token in the vocabulary.
    double[][] rms_att_weight; /// Layer normalization applied before multi-head attention.
    double[][] rms_ffn_weight; /// Layer normalization applied before feed-forward.
    double[] rms_final_weight; /// Layer normalization applied as final stage.
    double[][] freq_cis_real; /// Lookup table for Rotary Positional Encoding (RoPE).
    double[][] freq_cis_imag; /// Lookup table for Rotary Positional Encoding (RoPE).
    double[][][] wq; /// Learnt matrix wk used to compute query vector for a given layer.
    double[][][] wv; /// Learnt matrix wk used to compute value vector for a given layer.
    double[][][] wk; /// Learnt matrix wk used to compute key vector for a given layer.
    double[][][] wo; /// Learnt matrix wo applied after multi-head attention.
    double[][][] w1; /// Learnt matrix w1 used in feed-forward stage.
    double[][][] w2; /// Learnt matrix w2 used in feed-forward stage.
    double[][][] w3; /// Learnt matrix w3 used in feed-forward stage.
    string[] vocab; /// Strings containing the word for each token in the vocabulary.

    double[,,] key_cache; /// Copies of the key/value vectors for all preceding tokens in the sequence.
    double[,,] value_cache; /// Avoids repeatedly recomputing the vectors for every layer.

    double[] x; /// Accumulated result vector across transformer layers.
    double[] xb; /// Intermediate results for multi-head attention.
    double[] xb2; /// Intermediate results for multi-head attention.
    double[] hb; /// Intermediate results for feed-forward stage.
    double[] hb2; /// Intermediate results for feed-forward stage.
    double[] q; /// Query vector used in transformer blocks.
    double[] k; /// Key vector used in transformer blocks.
    double[] v; /// Value vector used in transformer blocks.
    double[] attention; /// Output attention for each layer.
    double[] logits; /// Output logits from model.

    /// <summary>
    /// Reads a three-dimensional array of single precision 32-bit floating point values
    /// from the given file/reader, in row-major order, converted to double precision.
    /// These usually represent an array of matrices. Each matrix is assumed to be of
    /// size <paramref name="dim2"/> x <paramref name="dim3"/>, with <paramref name="dim1"/>
    /// matrices total read from the file.
    /// </summary>
    /// <param name="reader">Reader for file to read from</param>
    /// <param name="dim1">Number of matrices to read</param>
    /// <param name="dim2">Number of rows in each matrix</param>
    /// <param name="dim3">Number of columns in each matrix</param>
    /// <returns>Matrix loaded from file</returns>
    double[][][] Read3D(BinaryReader reader, int dim1, int dim2, int dim3)
    {
        var output = new double[dim1][][];
        for (int i = 0; i < dim1; i++)
            output[i] = Read2D(reader, dim2, dim3);
        return output;
    }

    /// <summary>
    /// Reads a two-dimensional array of single precision 32-bit floating point values
    /// from the given file/reader, in row-major order, converted to double precision.
    /// This two-dimensional array is usually a matrix. The matrix is assumed to be of
    /// size <paramref name="dim1"/> x <paramref name="dim2"/>.
    /// </summary>
    /// <param name="reader">Reader for file to read from</param>
    /// <param name="dim1">Number of rows in the matrix</param>
    /// <param name="dim2">Number of columns in the matrix</param>
    /// <returns>Matrix loaded from file</returns>
    double[][] Read2D(BinaryReader reader, int dim1, int dim2)
    {
        var output = new double[dim1][];
        for (int i = 0; i < dim1; i++)
            output[i] = Read1D(reader, dim2);
        return output;
    }

    /// <summary>
    /// Reads an array of single precision 32-bit floating point values from the given
    /// file/reader, converted to double precision. The array is assumed to be of size
    /// <paramref name="dim"/>.
    /// </summary>
    /// <param name="reader">Reader for file to read from</param>
    /// <param name="dim">Size of the array</param>
    /// <returns>Array of values loaded from file</returns>
    double[] Read1D(BinaryReader reader, int dim)
    {
        var output = new double[dim];
        for (int i = 0; i < dim; i++)
            output[i] = reader.ReadSingle();
        return output;
    }

    /// <summary>
    /// Constructor for the transformer class.
    /// Loads the specified model into memory, and pre-allocates storage for inference.
    /// </summary>
    /// <param name="modelfile">Path to the file containing model parameters</param>
    /// <param name="tokenizerfile">Path to the file containing token strings</param>
    public Transformer(string modelfile, string tokenizerfile)
    {
        // Read the entire model into memory. The file itself is not easily interpretable, but
        // the structure mostly follows version 1 from llama2.c:
        // https://github.com/karpathy/llama2.c/blob/master/export.py
        using (BinaryReader reader = new BinaryReader(File.Open(modelfile, FileMode.Open)))
        {
            dim = reader.ReadInt32();
            hidden_dim = reader.ReadInt32();
            n_layers = reader.ReadInt32();
            n_heads = reader.ReadInt32();
            var n_kv_heads = reader.ReadInt32();
            vocab_size = reader.ReadInt32();
            seq_len = reader.ReadInt32();

            token_embedding = Read2D(reader, vocab_size, dim);
            rms_att_weight = Read2D(reader, n_layers, dim);
            wq = Read3D(reader, n_layers, dim, dim);
            wk = Read3D(reader, n_layers, dim, dim);
            wv = Read3D(reader, n_layers, dim, dim);
            wo = Read3D(reader, n_layers, dim, dim);
            rms_ffn_weight = Read2D(reader, n_layers, dim);
            w1 = Read3D(reader, n_layers, hidden_dim, dim);
            w2 = Read3D(reader, n_layers, dim, hidden_dim);
            w3 = Read3D(reader, n_layers, hidden_dim, dim);
            rms_final_weight = Read1D(reader, dim);
            head_size = dim / n_heads;
            freq_cis_real = Read2D(reader, seq_len, head_size / 2);
            freq_cis_imag = Read2D(reader, seq_len, head_size / 2);
        }

        // The string/characters for each token are stored in a separate file.
        using (BinaryReader reader = new BinaryReader(File.Open(tokenizerfile, FileMode.Open)))
        {
            vocab = new string[vocab_size];
            for (int i = 0; i < vocab_size; i++)
            {
                var len = reader.ReadInt32();
                var bytes = reader.ReadBytes(len);
                var chars = Encoding.Default.GetChars(bytes);
                vocab[i] = new String(chars);
            }
        }

        // Pre-allocate storage for intermediate and result vectors.
        // We allocate these once at startup to avoid repeatedly reallocating during
        // inference, as this would significantly impact execution speed.
        x = new double[dim];
        xb = new double[dim];
        xb2 = new double[dim];
        hb = new double[hidden_dim];
        hb2 = new double[hidden_dim];
        q = new double[dim];
        k = new double[dim];
        v = new double[dim];
        attention = new double[seq_len];
        logits = new double[vocab_size];

        // For the key/value stage, we allocate sufficient storage to store the key/value
        // vectors for all preceding tokens up to the maximum sequence length.
        key_cache = new double[n_layers, seq_len, dim];
        value_cache = new double[n_layers, seq_len, dim];
    }

    /// <summary>
    /// Performs root mean square (RMS) layer normalization on an input vector.
    /// If you are interested, you can read the paper at https://arxiv.org/pdf/1910.07467.
    /// To apply RMS layer normalization, we compute:
    ///
    ///   rms = sqrt(mean(input ^ 2))
    ///   output[i] = (output[i] / rms) * weights[i]
    ///
    /// Weights is a learnt parameter that is computed during training.
    ///
    /// Compared to layer normalization, RMSNorm is more computationally efficient,
    /// and helps stabilize the training of transformer-based models.
    /// </summary>
    /// <param name="output">Output vector (array) to store the result in</param>
    /// <param name="x">Input vector (array) that is going to be normalized</param>
    /// <param name="weight">Learned weights (array) of scaling factors</param>
    void rmsnorm(double[] output, double[] x, double[] weight)
    {
        // Calculate the mean of all squared values, aka the root mean square (RMS).
        double ss = 0.0;
        for (int j = 0; j < x.Length; j++)
        {
            var sqr = x[j] * x[j];
            ss += sqr;
        }
        ss /= x.Length;

        // Avoid division by zero by adding a very small number (aka epsilon).
        ss += 1e-5;

        // Compute rms = sqrt(mean(input ^ 2)). We take the reciprocal of this value to
        // avoid a division in the loop, because (x * (1.0 / y)) ~= (x / y), and
        // multiplication is faster than division.
        ss = 1.0 / Math.Sqrt(ss);

        // Compute (output[i] / rms) * weights[i].
        for (int j = 0; j < x.Length; j++)
            output[j] = weight[j] * (ss * x[j]);
    }

    /// <summary>
    /// Applies the softmax function to the given input vector (array).
    /// Softmax is a function that transforms a vector into a probability distribution,
    /// ranging from 0 to 1.
    ///
    /// Softmax is computed as:
    ///   softmax(xi) = exp(xi) / sum(exp(xj))
    ///
    /// </summary>
    /// <param name="output">Array to store the resulting probabilities</param>
    /// <param name="input">Array of input values (logits)</param>
    /// <param name="pos">Last valid index in the arrays (array length - 1)</param>
    void softmax(double[] output, double[] input, int pos)
    {
        // Find the maximum value in the input array for numerical stability.
        // This prevents overflow by keeping the exponent portion of the
        // floating-point value (power of 2) smaller by subtracing the maximum
        // value before the exp() in the next section.
        double max_val = input[0];
        for (int i = 1; i <= pos; i++)
            if (input[i] > max_val)
                max_val = input[i];

        // Compute exp(x - max_val) for each vector element, and accumulate.
        double sum = 0.0;
        for (int i = 0; i <= pos; i++)
        {
            output[i] = Math.Exp(input[i] - max_val);
            sum += output[i];
        }

        // Normalize by dividing by the sum, ensuring that all output values
        // are between 0 and 1.
        for (int i = 0; i <= pos; i++)
            output[i] /= sum;
    }

    /// <summary>
    /// Computes the product of x and y by multiplying the matrix w by the vector x.
    /// Given matrix w with size m x n, the input vector should be of length m, and
    /// the output vector size n.
    /// </summary>
    /// <param name="output">Output vector of size m</param>
    /// <param name="x">Input vector of size n</param>
    /// <param name="w">Weight matrix of size m x n</param>
    void matmul(double[] output, double[] x, double[][] w)
    {
        for (int i = 0; i < output.Length; i++)
        {
            // Compute dot product of row i of w with vector x.
            double val = 0.0;
            for (int j = 0; j < x.Length; j++)
                val += w[i][j] * x[j];

            // And store to output row i.
            output[i] = val;
        }
    }

    /// <summary>
    /// Adds two vectors together. The first vector is updated in-place.
    /// Both vectors should be of the dimension (array size).
    /// </summary>
    /// <param name="lhs">Left hand side of the expression</param>
    /// <param name="rhs">Right hand size of the expression</param>
    void accum(double[] lhs, double[] rhs)
    {
        for (int i = 0; i < lhs.Length; i++)
            lhs[i] += rhs[i];
    }

    /// <summary>
    /// Copies the contents of one vector to another.
    /// Both vectors should be of the dimension (array size).
    /// </summary>
    /// <param name="dst">Destination vector</param>
    /// <param name="src"></param>
    void copy(double[] dst, double[] src)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i];
    }

    /// <summary>
    /// Processes a single token through the transformer model to generate the logits
    /// (probabilities for each token) of the next token in the sequence.
    ///
    /// This is a decoder-only transformer, similar to what is found in GPT models.
    ///
    /// </summary>
    /// <param name="next_token">ID of the token to predict the next token for.</param>
    /// <param name="pos">The position of this token in the sequence.</param>
    void transformer(int next_token, int pos)
    {
        // Get the token embedding for the input token.
        copy(x, token_embedding[next_token]);

        // Process each layer of the transformer model.
        for (int layer = 0; layer < n_layers; layer++)
        {
            // Apply layer normalization to the current vector (embedding) before attention.
            rmsnorm(xb, x, rms_att_weight[layer]);

            // Generate query, key and value vectors by multiplying the current vector by
            // the corresponding query, key and value matrices for this layer.
            // q = query, representation of the current token.
            // k = key, which we match our query against to determine attention.
            // v = value, which contains information about the token.
            matmul(q, xb, wq[layer]);
            matmul(k, xb, wk[layer]);
            matmul(v, xb, wv[layer]);

            // Apply Rotary Position Embedding(RoPE) to query and key vectors.
            // RoPE injects positional information through rotation in 2D subspaces in a
            // way that also captures the relative positioning between tokens.
            for (int head = 0; head < n_heads; head++)
            {
                // See the next section for more detail on each "head" of multi-head attention.
                // We increment by 2 because we're treating the vector elements as 2D coordinates.
                for (int i = 0; i < head_size; i += 2)
                {
                    // Obtain rotation parameters for this position.
                    // Instead of simply adding embedded positional data to the vectors, RoPE
                    // performs 2D rotation by an angle that depends on both the position in
                    // the sequence (pos), and the dimension index in the vector (i). These
                    // rotational values are part of the model, therefore we simply need to
                    // load them from the precomputed arrays.
                    var fcr = freq_cis_real[pos][i / 2];
                    var fci = freq_cis_imag[pos][i / 2];

                    // Apply 2D rotation to both vectors.
                    var q0 = q[head * head_size + i];
                    var q1 = q[head * head_size + i + 1];
                    var k0 = k[head * head_size + i];
                    var k1 = k[head * head_size + i + 1];
                    
                    q[head * head_size + i] = q0 * fcr - q1 * fci;
                    q[head * head_size + i + 1] = q0 * fci + q1 * fcr;
                    k[head * head_size + i] = k0 * fcr - k1 * fci;
                    k[head * head_size + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            // Store a copy of the key and value vectors for this token, which will be
            // used for computing attention with future tokens, avoiding redundant computation.
            for (int i = 0; i < dim; i++)
            {
                key_cache[layer, pos, i]   = k[i];
                value_cache[layer, pos, i] = v[i];
            }

            // This next block of code implements multi-head attention. Each "head" can focus on different aspects
            // of the input sequence, e.g. grammar and topics, and has its own query, key, and value vectors.
            //
            // We start by iterating over all heads:
            //
            for (int head = 0; head < n_heads; head++)
            {
                // All tokens currently output in the sequence are considered to determine the next token.
                // The <= in this loop is not an error, pos contains the index of the token we are currently computing,
                // which was written to the key/value cache in the loop above.
                for (int token_pos = 0; token_pos <= pos; token_pos++)
                {
                    // We want to determine how much attention should be given to each token (soft attention).
                    // We do this by computing the dot product between the query and key vectors.
                    //
                    // head_size contains the size of each head, which are concatenated together in the query, key and
                    // value vectors. This isn't relevant to our implementation here, but for larger models being able
                    // to compute the attention for each head in parallel significantly improves performance.
                    //
                    // The total attention score is scaled by the square root of the head size to improve gradient stability.
                    //
                    var score = 0.0;
                    for (int i = 0; i < head_size; i++)
                        score += q[head * head_size + i] * key_cache[layer, token_pos, head * head_size + i];
                    score /= Math.Sqrt(head_size);
                    attention[token_pos] = score;
                }

                // Convert attention scores to a list of probabilities between 0 and 1 using the softmax function.
                softmax(attention, attention, pos);

                // Recompute the vector for the next token by weighting the value vectors by the probabilities
                // for each token. We utilize the value cache updated above to avoid recomputing the value
                // vectors for all the previous tokens.
                for (int i = 0; i < head_size; i++)
                {
                    double val = 0.0;
                    for (int token = 0; token <= pos; token++)
                        val += attention[token] * value_cache[layer, token, head * head_size + i];
                    xb[head * head_size + i] = val;
                }
            }

            // Project concatenated attention outputs with the output matrix to produce final attention.
            matmul(xb2, xb, wo[layer]);

            // The residual connection creates another path for data from one layer to reach later layers
            // by skipping intermediate layers. This is achieved by simply adding the current layer's
            // output representation to the embedding that we feed into the next layer.
            accum(x, xb2);

            // Apply layer normalization before the final feed-forward neural network.
            rmsnorm(xb, x, rms_ffn_weight[layer]);

            // Our two linear projections (feed forward stage). Simple matrix multiplications.
            matmul(hb, xb, w1[layer]);
            matmul(hb2, xb, w3[layer]);

            // Apply our activation function: Sigmoid Linear Unit (SilU)
            // SilU is computed as silu(x) = x*σ(x).
            // σ(x) is the logistic sigmoid, or σ(x) = 1 / 1 + exp(-x).
            for (int i = 0; i < hidden_dim; i++)
                hb[i] = hb[i] * (1.0 / (1.0 + Math.Exp(-hb[i])));

            // Gated activation only applies the activation function to w1.
            for (int i = 0; i < hidden_dim; i++)
                hb[i] = hb[i] * hb2[i];

            // Final linear projection of the feed forward stage.
            matmul(xb, hb, w2[layer]);

            // Accumulate residual (skip) connection.
            accum(x, xb);
        }

        // Final layer normalization.
        rmsnorm(x, x, rms_final_weight);

        // Project to the vocabulary (token embedding), returning logits for each token.
        // Each element represents the probability for this token.
        matmul(logits, x, token_embedding);
    }

    /// <summary>
    /// Returns the index of element containing the the maximum value in a given vector,
    /// also known as "argmax".
    /// </summary>
    /// <param name="v">Input vector</param>
    /// <returns>Index of element with the largest value</returns>
    int argmax(double[] v)
    {
        // return argmax of v in elements 0..n
        int max_i = 0;
        double max_p = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max_p)
            {
                max_i = i;
                max_p = v[i];
            }
        return max_i;
    }

    /// <summary>
    /// Applies the transformer model to predict tokens until the end of sequence or maximum sequence
    /// length is reached, whichever is shorter. For the model provided with the assignment, this
    /// results in a short story.
    /// </summary>
    public void tellStory()
    {
        // Start by predicting the next token for the beginning-of-sequence token (BOS, 1).
        int token = 1;
        int pos = 0;

        // Continue until we reach another beginning-of-sequence token, or the maximum length.
        while (pos < seq_len)
        {
            // Applies our model to predict the next token in the sequence. This returns a vector
            // of logits, i.e. the probabilities for each token being the next token in the sequence.
            transformer(token, pos);

            // We use argmax to find the element with the greatest probability.
            // Other strategies can be used here to influence the resulting test, for example, you could
            // randomly select from the top N tokens instead of always taking the "most likely" token.
            // This makes the output more "human-like", however it will vary from run-to-run based on
            // the seed given to the PRNG. For these implementations, we choose the "most likely" token
            // so the output is deterministic, which simplifies testing as there is only one "correct"
            // output for the program for a given model.
            var next = argmax(logits);

            // Is this the end of the sequence? End if so.
            // We don't print the BOS/EOS token.
            if (next == 1)
                break;

            // Display the selected token.
            Console.Write(vocab[next]);

            // Repeat the process again, using this just-predicted token to compute the next token.
            token = next;
            pos++;
        }
    }
}