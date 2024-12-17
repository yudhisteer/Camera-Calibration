using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Initialize the visualizer with checkerboard size
            var checkerboardSize = new Size(7, 7); // Adjust based on your checkerboard
            var visualizer = new AlignmentVisualizer(checkerboardSize);

            // Load test image
            Console.WriteLine("Loading checkerboard image...");
            string imagePath = "C:\\Users\\v-ychintaram\\OneDrive - Microsoft\\Desktop\\__Project__\\Camera-Calibration\\csharp\\Calibration\\reference_screen.png";
            Mat image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);

            if (image.IsEmpty)
            {
                throw new Exception($"Failed to load image from {imagePath}");
            }

            // Find checkerboard corners
            Console.WriteLine("Finding checkerboard corners...");
            VectorOfPointF corners = new VectorOfPointF();
            bool found = CvInvoke.FindChessboardCorners(image, checkerboardSize, corners);

            if (!found)
            {
                throw new Exception("Checkerboard pattern not found in the image");
            }

            // Refine corner detection
            Console.WriteLine("Refining corner detection...");
            MCvTermCriteria criteria = new MCvTermCriteria(30, 0.001);
            CvInvoke.CornerSubPix(
                image,
                corners,
                new Size(11, 11),    // Search window size
                new Size(-1, -1),    // Zero zone (-1 = no zero zone)
                criteria
            );

            // Get the corner points as an array
            PointF[] cornerPoints = corners.ToArray();

            // Calculate some example metrics (replace with your actual measurements)
            var metrics = new Dictionary<string, float>
            {
                { "pattern_width", 200.0f },
                { "pattern_height", 200.0f },
                { "width_ratio", 0.95f },
                { "height_ratio", 0.98f },
                { "horizontal_ratio", 1.02f },
                { "vertical_ratio", 0.99f }
            };

            // Visualize the results
            Console.WriteLine("Visualizing results...");

            // Draw and save corners visualization
            visualizer.DrawCorners(image, cornerPoints, "Detected_Corners");

            // Draw and save bounds visualization
            visualizer.DrawBounds(image, cornerPoints, metrics, "Pattern_Bounds");

            Console.WriteLine("Processing complete. Check the output files.");

            // Proper cleanup
            corners.Dispose();
            image.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}