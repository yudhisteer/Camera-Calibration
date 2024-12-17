using System;
using System.ComponentModel;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

public class AlignmentChecker
{
    private Size checkerboardSize;
    private double maxRotationError;
    private double maxScaleDifference;
    private AlignmentVisualizer visualizer;
    private int imageWidth;
    private int imageHeight;

    public AlignmentChecker(Size? checkerboardSize = null, double maxRotationError = 5.0, double maxScaleDifference = 0.1)
    {
        this.checkerboardSize = checkerboardSize ?? new Size(7, 7);
        this.maxRotationError = maxRotationError;
        this.maxScaleDifference = maxScaleDifference;
        this.visualizer = new AlignmentVisualizer(this.checkerboardSize);
    }

    public (PointF[] corners, Mat image) FindCorners(string imagePath)
    {
        // Load image
        Mat image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
        if (image.IsEmpty)
        {
            throw new ArgumentException("Could not load image");
        }

        imageHeight = image.Height;
        imageWidth = image.Width;

        // Save original image
        image.Save("Original_Image.png");
        Console.WriteLine("Saved original image to Original_Image.png");

        // Find corners
        VectorOfPointF corners = new VectorOfPointF();
        bool found = CvInvoke.FindChessboardCorners(image, checkerboardSize, corners);

        if (!found)
        {
            throw new InvalidOperationException("Could not find checkerboard corners in image");
        }

        // Refine corner positions
        MCvTermCriteria criteria = new MCvTermCriteria(30, 0.001);
        CvInvoke.CornerSubPix(image, corners, new Size(11, 11), new Size(-1, -1), criteria);

        // Visualize detected corners
        PointF[] cornerPoints = corners.ToArray();
        visualizer.DrawCorners(image, cornerPoints, "Detected_Corners");

        corners.Dispose();
        return (cornerPoints, image);
    }

    public Dictionary<string, float> CalculatePatternMetrics(PointF[] corners, Mat image)
    {
        var metrics = CalculateBasicMetrics(corners, image);
        visualizer.DrawBounds(image, corners, metrics, "Pattern_Bounds");
        return metrics;
    }

    private Dictionary<string, float> CalculateBasicMetrics(PointF[] corners, Mat image)
    {
        var metrics = new Dictionary<string, float>
        {
            { "min_x", corners.Min(c => c.X) },
            { "max_x", corners.Max(c => c.X) },
            { "min_y", corners.Min(c => c.Y) },
            { "max_y", corners.Max(c => c.Y) }
        };

        // Calculate distances and dimensions
        metrics.Add("left_distance", metrics["min_x"]);
        metrics.Add("right_distance", image.Width - metrics["max_x"]);
        metrics.Add("top_distance", metrics["min_y"]);
        metrics.Add("bottom_distance", image.Height - metrics["max_y"]);
        metrics.Add("pattern_width", metrics["max_x"] - metrics["min_x"]);
        metrics.Add("pattern_height", metrics["max_y"] - metrics["min_y"]);

        // Calculate ratios
        metrics.Add("width_ratio", metrics["pattern_width"] / image.Width);
        metrics.Add("height_ratio", metrics["pattern_height"] / image.Height);
        metrics.Add("horizontal_ratio", metrics["left_distance"] / (metrics["left_distance"] + metrics["right_distance"]));
        metrics.Add("vertical_ratio", metrics["top_distance"] / (metrics["top_distance"] + metrics["bottom_distance"]));
        metrics.Add("image_width", image.Width);
        metrics.Add("image_height", image.Height);

        return metrics;
    }

    public Dictionary<string, object> CheckAlignment(string referenceImagePath, string testImagePath)
    {
        // Find corners in both images
        var (refCorners, refImage) = FindCorners(referenceImagePath);
        var (testCorners, testImage) = FindCorners(testImagePath);

        // Calculate metrics
        var refMetrics = CalculatePatternMetrics(refCorners, refImage);
        var testMetrics = CalculatePatternMetrics(testCorners, testImage);

        // Calculate differences
        var differences = CalculateDifferences(refCorners, testCorners, refMetrics, testMetrics);

        // Check alignment
        var alignmentStatus = CheckAlignmentStatus(differences);


        // Add border check for test image
        var borderStatus = CheckScreenBorders(testImage);

        // Update alignment status to include border check
        alignmentStatus["no_screen_borders"] = !Convert.ToBoolean(borderStatus["has_screen_borders"]);

        // Update print results call
        PrintAlignmentResults(differences, alignmentStatus, refMetrics, testMetrics, borderStatus);


        // Cleanup
        refImage.Dispose();
        testImage.Dispose();

        return new Dictionary<string, object>
        {
            { "differences", differences },
            { "alignment_status", alignmentStatus },
            { "ref_metrics", refMetrics },
            { "test_metrics", testMetrics },
            { "border_status", borderStatus }
        };
    }

    private Dictionary<string, double> CalculateDifferences(PointF[] refCorners, PointF[] testCorners,
    Dictionary<string, float> refMetrics, Dictionary<string, float> testMetrics)
    {
        double horizontalDiff = Math.Abs(refMetrics["horizontal_ratio"] - testMetrics["horizontal_ratio"]);
        double verticalDiff = Math.Abs(refMetrics["vertical_ratio"] - testMetrics["vertical_ratio"]);
        double widthRatioDiff = Math.Abs(refMetrics["width_ratio"] - testMetrics["width_ratio"]);
        double heightRatioDiff = Math.Abs(refMetrics["height_ratio"] - testMetrics["height_ratio"]);

        // Calculate rotation
        var refTop = refCorners.Take(checkerboardSize.Width).ToArray();
        var testTop = testCorners.Take(checkerboardSize.Width).ToArray();

        // Calculate vectors
        float refDx = refTop.Last().X - refTop[0].X;
        float refDy = refTop.Last().Y - refTop[0].Y;
        float testDx = testTop.Last().X - testTop[0].X;
        float testDy = testTop.Last().Y - testTop[0].Y;

        // Check if vertical variation is larger than horizontal for test vector
        if (Math.Abs(testDy) > Math.Abs(testDx))
        {
            // Swap dx and dy if detected vertically
            float temp = testDx;
            testDx = testDy;
            testDy = temp;
        }

        double refAngle = Math.Atan2(refDy, refDx);
        double testAngle = Math.Atan2(testDy, testDx);
        double rotationError = Math.Abs(refAngle - testAngle) * 180 / Math.PI;

        return new Dictionary<string, double>
    {
        { "horizontal_difference", horizontalDiff },
        { "vertical_difference", verticalDiff },
        { "width_ratio_difference", widthRatioDiff },
        { "height_ratio_difference", heightRatioDiff },
        { "rotation_error", rotationError }
    
        };
    }



    private Dictionary<string, object> CheckScreenBorders(Mat image, int threshold = 30)
    {
        //+------------------+Legend:
        //| TTTTTTTTTTTTTTTTTT | T = Top border(20 pixels)
        //| L                R | B = Bottom border(20 pixels)
        //| L                R | L = Left border(20 pixels)
        //| L          Image R | R = Right border(20 pixels)
        //| L                R |
        //| L                R | Each border is 20 pixels wide
        //| L                R |
        //| BBBBBBBBBBBBBBBBBB |
        //+------------------+

        int borderSize = 20;  // Number of pixels to check from each edge
        var borders = new Dictionary<string, double>();

        // Get edge regions and calculate mean intensities
        using (var topBorder = new Mat(image, new Rectangle(0, 0, image.Width, borderSize)))
        using (var bottomBorder = new Mat(image, new Rectangle(0, image.Height - borderSize, image.Width, borderSize)))
        using (var leftBorder = new Mat(image, new Rectangle(0, 0, borderSize, image.Height)))
        using (var rightBorder = new Mat(image, new Rectangle(image.Width - borderSize, 0, borderSize, image.Height)))
        {
            borders["top"] = CvInvoke.Mean(topBorder).V0;
            borders["bottom"] = CvInvoke.Mean(bottomBorder).V0;
            borders["left"] = CvInvoke.Mean(leftBorder).V0;
            borders["right"] = CvInvoke.Mean(rightBorder).V0;
        }

        var borderStatus = new Dictionary<string, object>();

        // Check if borders are too dark
        foreach (var border in borders)
        {
            borderStatus[$"{border.Key}_border_visible"] = border.Value < threshold;
            borderStatus[$"{border.Key}_intensity"] = border.Value;
        }

        // Overall status
        borderStatus["has_screen_borders"] = borders.Any(b => b.Value < threshold);

        return borderStatus;
    }

    private Dictionary<string, bool> CheckAlignmentStatus(Dictionary<string, double> differences)
    {
        const double maxPositionRatioDiff = 0.1;

        return new Dictionary<string, bool>
        {
            { "is_horizontal_aligned", differences["horizontal_difference"] <= maxPositionRatioDiff },
            { "is_vertical_aligned", differences["vertical_difference"] <= maxPositionRatioDiff },
            { "is_rotation_aligned", differences["rotation_error"] <= maxRotationError },
            { "is_scale_aligned", differences["width_ratio_difference"] <= maxScaleDifference &&
                                differences["height_ratio_difference"] <= maxScaleDifference }
        };
    }

    private void PrintAlignmentResults(Dictionary<string, double> differences, Dictionary<string, bool> alignmentStatus,
        Dictionary<string, float> refMetrics, Dictionary<string, float> testMetrics, Dictionary<string, object> borderStatus)
    {
        Console.WriteLine("\nAlignment Check Results:");

        Console.WriteLine("\nScreen Border Analysis:");

        //✗ (X)means a border was detected(intensity below threshold of 30)
        //✓ (√) means no border was detected(intensity above threshold of 30)

        //0 = completely black
        //255 = completely white

        foreach (var direction in new[] { "top", "bottom", "left", "right" })
        {
            Console.WriteLine($"{char.ToUpper(direction[0]) + direction.Substring(1)} border intensity: " +
                $"{Convert.ToDouble(borderStatus[$"{direction}_intensity"]):F1}");
            Console.WriteLine($"{char.ToUpper(direction[0]) + direction.Substring(1)} border visible: " +
                $"{(Convert.ToBoolean(borderStatus[$"{direction}_border_visible"]) ? "X" : "✓")}");
        }

        Console.WriteLine($"\nScreen Border Check: " +
            $"{(Convert.ToBoolean(borderStatus["has_screen_borders"]) ? "X" : "✓")}");

        bool isAlignedBorders = alignmentStatus.Values.All(v => v) && !Convert.ToBoolean(borderStatus["has_screen_borders"]);
        Console.WriteLine($"Keystone Check Status: {(isAlignedBorders ? "PASS" : "FAIL")}");

        Console.WriteLine("\nPosition Analysis:");
        Console.WriteLine($"Horizontal alignment difference: {differences["horizontal_difference"]:F3} " +
                        $"{(alignmentStatus["is_horizontal_aligned"] ? "✓" : "X")}");
        Console.WriteLine($"Vertical alignment difference: {differences["vertical_difference"]:F3} " +
                        $"{(alignmentStatus["is_vertical_aligned"] ? "✓" : "X")}");

        Console.WriteLine("\nScale Analysis:");
        Console.WriteLine($"Reference pattern/image width ratio: {refMetrics["width_ratio"]:F3}");
        Console.WriteLine($"Test pattern/image width ratio: {testMetrics["width_ratio"]:F3}");
        Console.WriteLine($"Width ratio difference: {differences["width_ratio_difference"]:F3}");
        Console.WriteLine($"Height ratio difference: {differences["height_ratio_difference"]:F3}");
        Console.WriteLine($"Scale alignment: {(alignmentStatus["is_scale_aligned"] ? "✓" : "X")}");

        Console.WriteLine($"\nRotation Error: {differences["rotation_error"]:F2}° " +
                        $"{(alignmentStatus["is_rotation_aligned"] ? "✓" : "X")}");

        bool isAligned = alignmentStatus.Values.All(v => v);
        Console.WriteLine($"\nOverall Status: {(isAligned ? "PASS" : "FAIL")}");
    }
}