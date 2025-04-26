open Transformer
open IO
open Decoders

let model53M = readModel "model.bin" "tokenizer.bin"

tellStory model53M mostLikely
