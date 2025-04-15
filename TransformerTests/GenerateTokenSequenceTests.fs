namespace TransformerFunctions

open Transformer
open Decoders
open Microsoft.VisualStudio.TestTools.UnitTesting
open TransformerTestsHelpers

[<TestClass>]
type GenerateTokenSequenceTests () =

    let model53M = readModel53M

    [<TestMethod>]
    member this.Test1 () =
        let actual = generateTokenSequence model53M mostLikely |> Seq.toArray
        let expected = [|" Once"; " upon"; " a"; " time"; ","; " there"; " was"; " a"; " little"; " girl"; " named"; " L"; "ily"; "."; " She"; " loved"; " to"; " play"; " outside"; " in"; " the"; " sun"; "sh"; "ine"; "."; " One"; " day"; ","; " she"; " saw"; " a"; " big"; ","; " red"; " ball"; " in"; " the"; " sky"; "."; " It"; " was"; " the"; " sun"; "!"; " She"; " thought"; " it"; " was"; " so"; " pretty"; "."; "\n"; "L"; "ily"; " wanted"; " to"; " play"; " with"; " the"; " ball"; ","; " but"; " it"; " was"; " too"; " high"; " up"; " in"; " the"; " sky"; "."; " She"; " tried"; " to"; " jump"; " and"; " reach"; " it"; ","; " but"; " she"; " couldn"; "'"; "t"; "."; " Then"; ","; " she"; " had"; " an"; " idea"; "."; " She"; " would"; " use"; " a"; " stick"; " to"; " knock"; " the"; " ball"; " down"; "."; "\n"; "L"; "ily"; " found"; " a"; " stick"; " and"; " tried"; " to"; " hit"; " the"; " ball"; "."; " But"; " the"; " stick"; " was"; " too"; " short"; "."; " She"; " tried"; " again"; " and"; " again"; ","; " but"; " she"; " couldn"; "'"; "t"; " reach"; " it"; "."; " She"; " felt"; " sad"; "."; "\n"; "S"; "ud"; "den"; "ly"; ","; " a"; " kind"; " man"; " came"; " by"; " and"; " saw"; " L"; "ily"; "."; " He"; " asked"; " her"; " what"; " was"; " wrong"; "."; " L"; "ily"; " told"; " him"; " about"; " the"; " ball"; "."; " The"; " man"; " smiled"; " and"; " said"; ","; " \""; "I"; " have"; " a"; " useful"; " idea"; "!\""; " He"; " took"; " out"; " a"; " long"; " stick"; " and"; " used"; " it"; " to"; " knock"; " the"; " ball"; " down"; "."; " L"; "ily"; " was"; " so"; " happy"; "!"; " She"; " thank"; "ed"; " the"; " man"; " and"; " they"; " played"; " together"; " in"; " the"; " sun"; "sh"; "ine"; "."|]
        CollectionAssert.AreEqual(expected, actual)