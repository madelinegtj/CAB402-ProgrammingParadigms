namespace AttentionFunctions

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open Attention

[<TestClass>]
type AttentionForOneHeadTests() =

    let tolerance = 0.0

    let lookup1 i j = 
        let result = Random(i*1000 + j).NextDouble()
        result

    let lookup2 i j = 
        let result = Random(1000001 + i*997 + j).NextDouble()
        result
        
    [<TestMethod>]
    member this.Test1 () =          
        let tokenPosition = 1
        let query = [|0.4176901839; 0.320554296; 0.9149600104; 0.00802356084; 0.3437807155; 0.09508445416; 0.5081647022|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5061052923929366; 0.4283032446772603; 0.5509559207712025; 0.4731538730555261; 0.5958065491494683; 0.11823186333860124; 0.6406571775277341 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test2 () =
        let tokenPosition = 9
        let query = [|0.8158937302; 0.6311358489; 0.2242130826; 0.183478764; 0.6342530981|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5588057304291678; 0.512413797479954; 0.486982631364851; 0.5572644258582198; 0.5318332597431168 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test3 () =
        let tokenPosition = 5
        let query = [|0.8980199275; 0.4297310773; 0.0072189453; 0.6313039868; 0.9941349937; 0.0248348585|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5957691775117722; 0.5283156043839634; 0.4454350682293866; 0.5731662327622293; 0.4902856966076524; 0.47715378045281903 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test4 () =
        let tokenPosition = 7
        let query = [|0.8320466357|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5924734231563575 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test5 () =
        let tokenPosition = 3
        let query = [|0.7843547772; 0.6829007098; 0.4662664892; 0.675559153; 0.1279152177; 0.638195246; 0.7016369838; 0.01032251887|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.36946105173990007; 0.5982982738995253; 0.41431168011816594; 0.6431489022777911; 0.45916230849643175; 0.4278346801205112; 0.5040129368746976; 0.472685308498777 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test6 () =
        let tokenPosition = 4
        let query = [|0.09593777366; 0.6088603477; 0.107069242; 0.8119318486; 0.7855997201|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5308650521012079; 0.5890317524842259; 0.3074955577379306; 0.6338823808624917; 0.3523461861161964 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test7 () =
        let tokenPosition = 8
        let query = [|0.1049690313; 0.9176103819; 0.7550292031; 0.5560869671; 0.4056146038; 0.4915775173; 0.1182806211; 0.9665775022|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5717323478833936; 0.5100379857565919; 0.5066095349418578; 0.5548886141348577; 0.5514601633201236; 0.3621289685074534; 0.5963107916983894; 0.40697959688571916 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test8 () =
        let tokenPosition = 6
        let query = [|0.1763878839; 0.529698882; 0.4893993278; 0.295511229|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5954498617517499; 0.5038581935082578; 0.47117569208863735; 0.5487088218865236 |]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)

    [<TestMethod>]
    member this.Test9 () =    
        let tokenPosition = 7
        let query = [|0.3399223156; 0.8319186773; 0.6710281467; 0.008163234574; 0.634748524; 0.4052365069; 0.7273185991; 0.5628254333|]
        let actual = attentionForOneHead lookup1 lookup2 tokenPosition query
        actual |> Array.iter (fun f -> printf "%s; " (f.ToString("R")))
        let expected = [| 0.5979413100864399; 0.4312104646970629; 0.523876737066186; 0.4760610930753288; 0.5687273654444518; 0.40955304424273736; 0.6135779938227176; 0.4544036726210032|]
        Assert.AreEqual<int>(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.AreEqual(expected.[i], actual.[i], tolerance)