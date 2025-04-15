
namespace CSharpTransformer
{
    internal class Test
    {
        static void Main(string[] args)
        {
            var transformer = new Transformer("model.bin", "tokenizer.bin");

            transformer.tellStory();
        }
    }
}
