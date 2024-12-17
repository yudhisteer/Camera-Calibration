using Emgu.CV;
using System;
using System.Drawing;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Initialize the checker with checkerboard size and thresholds
            var checker = new AlignmentChecker(
                checkerboardSize: new Size(7, 7),
                maxRotationError: 5.0,
                maxScaleDifference: 0.06
            );

            // Define image paths
            string referenceImagePath = "C:\\Users\\v-ychintaram\\OneDrive - Microsoft\\Desktop\\__Project__\\Camera-Calibration\\csharp\\Calibration\\reference_screen.png";
            string testImagePath = "C:\\Users\\v-ychintaram\\OneDrive - Microsoft\\Desktop\\__Project__\\Camera-Calibration\\csharp\\Calibration\\rfc_1.jpg";

            // Run the alignment check
            var results = checker.CheckAlignment(referenceImagePath, testImagePath);

            Console.WriteLine("Processing complete. Check the output files.");
            // Keep windows open until a key is pressed
            Console.WriteLine("Press any key to close all windows...");
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
            Console.ReadKey();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}