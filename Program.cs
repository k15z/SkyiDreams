using System;
using System.IO;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

namespace SkyiDreams
{
    /// <summary>
    /// SkyiDreams uses an autoencoder network to learn the style of images and 
    /// dream up new ones. Just put some pictures in the memories folder - Skyi 
    /// will dream up some awesome pictures and put them in the dreams folder.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            MathNet.Numerics.Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            List<Vector<double>> data = new List<Vector<double>>();
            foreach (string file in Directory.GetFiles("memories"))
            {
                Console.WriteLine("Loading " + file + "...");
                Bitmap image = new Bitmap(Image.FromFile(file), new Size(400, 300));
                double[] raw = new double[400 * 300 * 3];
                for (int x = 0; x < 400; x++)
                    for (int y = 0; y < 300; y++)
                    {
                        raw[(y * 400 + x) * 3 + 0] = image.GetPixel(x, y).R / 125.0 - 1.0;
                        raw[(y * 400 + x) * 3 + 1] = image.GetPixel(x, y).G / 125.0 - 1.0;
                        raw[(y * 400 + x) * 3 + 2] = image.GetPixel(x, y).B / 125.0 - 1.0;
                    }
                data.Add(Vector<double>.Build.DenseOfArray(raw));
            }

            NeuralNetwork n1 = new NeuralNetwork(400 * 300 * 3, 10);
            NeuralNetwork n2 = new NeuralNetwork(10, 400 * 300 * 3);
            for (int epoch = 0; epoch < 100; epoch++)
            {
                Console.WriteLine("Epoch " + epoch + "...");
                double error = 0.0;
                foreach (Vector<double> vec in data)
                {
                    var result = n2.Feed(n1.Feed(vec));
                    n1.Train(n2.Train(vec.Subtract(result)));
                    error += vec.Subtract(result).SumMagnitudes();
                }
                Console.WriteLine("  error: " + error);

                {
                    Console.WriteLine("Dreaming...");
                    Random rand = new Random();
                    var raw = n2.Feed(n1.Feed(
                            data[rand.Next(data.Count)]
                                .Add(data[rand.Next(data.Count)])
                                .Divide(2)
                        )).ToArray<double>();
                    Bitmap image = new Bitmap(400, 300);
                    for (int x = 0; x < 400; x++)
                        for (int y = 0; y < 300; y++)
                        {
                            int r = (int)((raw[(y * 400 + x) * 3 + 0]+1) * 125);
                            int g = (int)((raw[(y * 400 + x) * 3 + 1]+1) * 125);
                            int b = (int)((raw[(y * 400 + x) * 3 + 2]+1) * 125);
                            r = Math.Min(255, Math.Max(0, r));
                            g = Math.Min(255, Math.Max(0, g));
                            b = Math.Min(255, Math.Max(0, b));
                            image.SetPixel(x, y, Color.FromArgb(r, g, b));
                        }
                    if (!Directory.Exists("dreams"))
                        Directory.CreateDirectory("dreams");
                    image.Save("dreams/dream."+epoch+".png");
                }
            }
        }
    }

    /// <summary>
    /// A feedforward neural network which uses the hyperbolic tangent as its 
    /// activation function and implements backpropagation with rmsprop.
    /// </summary>
    class NeuralNetwork
    {
        Vector<double> input;
        Vector<double> output;
        Matrix<double> weight;
        Matrix<double> rmsprop;

        public NeuralNetwork(int inputs, int outputs)
        {
            weight = Matrix<double>.Build.Random(rows: outputs, columns: inputs);
            rmsprop = Matrix<double>.Build.Random(rows: outputs, columns: inputs);
            rmsprop = rmsprop.Map(Math.Abs);
        }

        public Vector<double> Feed(Vector<double> input)
        {
            this.input = input;
            this.output = weight.Multiply(input);
            return output.Map(Math.Tanh);
        }

        public Vector<double> Train(Vector<double> error)
        {
            var deriv = error.PointwiseMultiply(output.Map(MathNet.Numerics.Differentiate.DerivativeFunc(Math.Tanh, 1)));

            double c = 0.99;
            var delta = deriv.OuterProduct(input);
            rmsprop = delta.PointwiseMultiply(delta).Multiply(1.0 - c).Add(rmsprop.Multiply(c));
            weight.Add(delta.PointwiseDivide(rmsprop.Map(Math.Sqrt)).Multiply(0.1), weight);

            return weight.LeftMultiply(deriv);
        }
    }
}
