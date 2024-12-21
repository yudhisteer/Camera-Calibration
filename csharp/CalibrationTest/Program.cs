using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace CalibrationTest
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // Initialize the checker
                var checker = new AlignmentChecker(
                    checkerboardSize: new Size(7, 7),
                    maxRotationError: 5.0,
                    maxScaleDifference: 0.06
                );

                // Image paths
                string referenceImagePath = "C:\\Users\\v-ychintaram\\OneDrive - Microsoft\\Desktop\\__Project__\\Camera-Calibration\\csharp\\Calibration\\reference_screen.png";
                string testImagePath = "C:\\Users\\v-ychintaram\\OneDrive - Microsoft\\Desktop\\__Project__\\Camera-Calibration\\csharp\\Calibration\\rfc_1.jpg";

                // Alignment check
                var results = checker.CheckAlignment(referenceImagePath, testImagePath);
                Console.WriteLine("Processing complete. Check the output files.");
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

    public class AlignmentVisualizer
    {
        private Size checkerboardSize;
        private const int DefaultFigureWidth = 1200;
        private const int DefaultFigureHeight = 800;

        public AlignmentVisualizer(Size? checkerboardSize = null)
        {
            this.checkerboardSize = checkerboardSize ?? new Size(7, 7);
        }

        public Mat DrawCorners(Mat image, PointF[] corners, string title = "Corners")
        {
            // Convert grayscale to RGB 
            Mat visImg = new Mat();
            CvInvoke.CvtColor(image, visImg, ColorConversion.Gray2Bgr);

            // Reshape corners and ensure consistent orientation
            var originalCorners = (PointF[])corners.Clone();
            var topRow = corners.Take(checkerboardSize.Width).ToArray();
            var firstPoint = topRow[0];
            var lastPoint = topRow[topRow.Length - 1];

            // Calculate if pattern is more vertical or horizontal
            float dx = Math.Abs(lastPoint.X - firstPoint.X);
            float dy = Math.Abs(lastPoint.Y - firstPoint.Y);

            // Adjust corner orientation if needed
            if (dy > dx)
            {
                var newCorners = ReshapeCorners(corners, true);
                corners = newCorners;
                topRow = corners.Take(checkerboardSize.Height).ToArray();
                firstPoint = topRow[0];
                lastPoint = topRow[topRow.Length - 1];

                if (firstPoint.X > lastPoint.X)
                {
                    corners = FlipCornersHorizontally(corners);
                }
            }
            else if (firstPoint.X > lastPoint.X)
            {
                corners = FlipCornersHorizontally(corners);
            }

            // Draw checkerboard pattern
            using (var vectorOfPoints = new VectorOfPointF(corners))
            {
                CvInvoke.DrawChessboardCorners(visImg, checkerboardSize, vectorOfPoints, true);
            }

            // Get corrected top row
            topRow = corners.Take(checkerboardSize.Width).ToArray();
            firstPoint = topRow[0];
            lastPoint = topRow[topRow.Length - 1];

            // Draw markers and connections
            DrawMarkers(visImg, firstPoint, lastPoint, corners);
            DrawConnections(visImg, corners);

            ShowImage(visImg, title);

            return visImg;
        }

        private void DrawMarkers(Mat visImg, PointF firstPoint, PointF lastPoint, PointF[] corners)
        {
            // Draw first corner (blue)
            Point firstCorner = Point.Round(new PointF(firstPoint.X, firstPoint.Y));
            CvInvoke.Circle(visImg, firstCorner, 10, new MCvScalar(255, 0, 0), 2);
            CvInvoke.PutText(visImg, "First", new Point(firstCorner.X - 20, firstCorner.Y - 10),
                FontFace.HersheySimplex, 0.5, new MCvScalar(255, 0, 0), 2);

            // Draw last corner (green)
            Point lastCorner = Point.Round(new PointF(lastPoint.X, lastPoint.Y));
            CvInvoke.Circle(visImg, lastCorner, 10, new MCvScalar(0, 255, 0), 2);
            CvInvoke.PutText(visImg, "Last", new Point(lastCorner.X - 20, lastCorner.Y - 10),
                FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 2);

            // Draw center point (yellow)
            PointF centerPoint = new PointF(
                corners.Average(p => p.X),
                corners.Average(p => p.Y)
            );
            Point center = Point.Round(centerPoint);
            CvInvoke.Circle(visImg, center, 10, new MCvScalar(0, 255, 255), 2);
            CvInvoke.PutText(visImg, "Center", new Point(center.X - 20, center.Y - 10),
                FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
        }

        private void DrawConnections(Mat visImg, PointF[] corners)
        {
            var cornersGrid = ReshapeToGrid(corners);
            var colors = new[]
            {
                new MCvScalar(255, 0, 0),   // Red
                new MCvScalar(0, 255, 0),   // Green
                new MCvScalar(0, 0, 255),   // Blue
                new MCvScalar(0, 255, 255), // Yellow
                new MCvScalar(255, 0, 255), // Magenta
                new MCvScalar(255, 255, 0)  // Cyan
            };

            for (int row = 0; row < checkerboardSize.Height; row++)
            {
                for (int col = 0; col < checkerboardSize.Width - 1; col++)
                {
                    Point startPoint = Point.Round(cornersGrid[row, col]);
                    Point endPoint = Point.Round(cornersGrid[row, col + 1]);
                    var color = colors[row % colors.Length];
                    CvInvoke.Line(visImg, startPoint, endPoint, color, 2);
                }
            }
        }

        public Mat DrawBounds(Mat image, PointF[] corners, Dictionary<string, float> metrics, string title = "Bounds")
        {
            Mat visImg = new Mat();
            CvInvoke.CvtColor(image, visImg, ColorConversion.Gray2Bgr);

            DrawBoundingBox(visImg, corners);
            DrawCenterLines(visImg, corners, image.Size);
            DrawMeasurements(visImg, metrics);

            ShowImage(visImg, title);

            return visImg;
        }

        private void DrawBoundingBox(Mat visImg, PointF[] corners)
        {
            var minX = (int)corners.Min(c => c.X);
            var maxX = (int)corners.Max(c => c.X);
            var minY = (int)corners.Min(c => c.Y);
            var maxY = (int)corners.Max(c => c.Y);

            var topLeft = new Point(minX, minY);
            var bottomRight = new Point(maxX, maxY);
            CvInvoke.Rectangle(visImg, new Rectangle(topLeft, new Size(maxX - minX, maxY - minY)),
                new MCvScalar(0, 255, 0), 2);
        }

        private void DrawCenterLines(Mat visImg, PointF[] corners, Size imageSize)
        {
            var cornersGrid = ReshapeToGrid(corners);
            int centerCol = checkerboardSize.Width / 2;
            int centerRow = checkerboardSize.Height / 2;

            // Pattern center lines
            Point topPoint = Point.Round(cornersGrid[0, centerCol]);
            Point bottomPoint = Point.Round(cornersGrid[checkerboardSize.Height - 1, centerCol]);
            CvInvoke.Line(visImg, topPoint, bottomPoint, new MCvScalar(0, 0, 255), 2);

            Point leftPoint = Point.Round(cornersGrid[centerRow, 0]);
            Point rightPoint = Point.Round(cornersGrid[centerRow, checkerboardSize.Width - 1]);
            CvInvoke.Line(visImg, leftPoint, rightPoint, new MCvScalar(0, 0, 255), 2);

            // Image center lines
            CvInvoke.Line(visImg, new Point(imageSize.Width / 2, 0),
                new Point(imageSize.Width / 2, imageSize.Height), new MCvScalar(255, 0, 0), 1);
            CvInvoke.Line(visImg, new Point(0, imageSize.Height / 2),
                new Point(imageSize.Width, imageSize.Height / 2), new MCvScalar(255, 0, 0), 1);
        }

        private void DrawMeasurements(Mat visImg, Dictionary<string, float> metrics)
        {
            string[] textLines = {
                $"Pattern Size: {metrics["pattern_width"]:F1} x {metrics["pattern_height"]:F1}",
                $"Width Ratio: {metrics["width_ratio"]:F3}",
                $"Height Ratio: {metrics["height_ratio"]:F3}",
                $"Horizontal Ratio: {metrics["horizontal_ratio"]:F3}",
                $"Vertical Ratio: {metrics["vertical_ratio"]:F3}"
            };

            for (int i = 0; i < textLines.Length; i++)
            {
                CvInvoke.PutText(visImg, textLines[i], new Point(10, 30 + i * 25),
                    FontFace.HersheySimplex, 0.6, new MCvScalar(255, 255, 255), 2);
            }
        }

        private PointF[] ReshapeCorners(PointF[] corners, bool transpose = false)
        {
            var grid = ReshapeToGrid(corners);
            if (transpose)
            {
                var transposed = new PointF[checkerboardSize.Width, checkerboardSize.Height];
                for (int i = 0; i < checkerboardSize.Height; i++)
                    for (int j = 0; j < checkerboardSize.Width; j++)
                        transposed[j, i] = grid[i, j];
                return FlattenGrid(transposed);
            }
            return corners;
        }

        private PointF[] FlipCornersHorizontally(PointF[] corners)
        {
            var grid = ReshapeToGrid(corners);
            var flipped = new PointF[checkerboardSize.Height, checkerboardSize.Width];
            for (int i = 0; i < checkerboardSize.Height; i++)
                for (int j = 0; j < checkerboardSize.Width; j++)
                    flipped[i, j] = grid[i, checkerboardSize.Width - 1 - j];
            return FlattenGrid(flipped);
        }

        private PointF[,] ReshapeToGrid(PointF[] corners)
        {
            var grid = new PointF[checkerboardSize.Height, checkerboardSize.Width];
            for (int i = 0; i < checkerboardSize.Height; i++)
                for (int j = 0; j < checkerboardSize.Width; j++)
                    grid[i, j] = corners[i * checkerboardSize.Width + j];
            return grid;
        }

        private PointF[] FlattenGrid(PointF[,] grid)
        {
            int rows = grid.GetLength(0);
            int cols = grid.GetLength(1);
            var flat = new PointF[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = grid[i, j];
            return flat;
        }

        private void ShowImage(Mat image, string title)
        {
            string filename = title.Replace(" ", "_") + ".png";
            image.Save(filename);
            Console.WriteLine($"Saved image to {filename}");
        }
    }

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
            Mat image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
            if (image.IsEmpty)
            {
                throw new ArgumentException("Could not load image");
            }

            imageHeight = image.Height;
            imageWidth = image.Width;

            VectorOfPointF corners = new VectorOfPointF();
            bool found = CvInvoke.FindChessboardCorners(image, checkerboardSize, corners);

            if (!found)
            {
                throw new InvalidOperationException("Could not find checkerboard corners in image");
            }

            MCvTermCriteria criteria = new MCvTermCriteria(30, 0.001);
            CvInvoke.CornerSubPix(image, corners, new Size(11, 11), new Size(-1, -1), criteria);

            PointF[] cornerPoints = corners.ToArray();
            string imageType = imagePath.Contains("reference") ? "Reference" : "Test";
            visualizer.DrawCorners(image, cornerPoints, $"{imageType}_Detected_Corners");

            corners.Dispose();
            return (cornerPoints, image);
        }

        public Dictionary<string, float> CalculatePatternMetrics(PointF[] corners, Mat image, string imageType)
        {
            var metrics = CalculateBasicMetrics(corners, image);
            visualizer.DrawBounds(image, corners, metrics, $"{imageType}_Pattern_Bounds");
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
            var refMetrics = CalculatePatternMetrics(refCorners, refImage, "Reference");
            var testMetrics = CalculatePatternMetrics(testCorners, testImage, "Test");

            // Calculate differences
            var differences = CalculateDifferences(refCorners, testCorners, refMetrics, testMetrics);

            // Check alignment
            var alignmentStatus = CheckAlignmentStatus(differences);

            // Check border for test image and print results
            var borderStatus = CheckScreenBorders(testImage);
            alignmentStatus["no_screen_borders"] = !Convert.ToBoolean(borderStatus["has_screen_borders"]);
            PrintAlignmentResults(differences, alignmentStatus, refMetrics, testMetrics, borderStatus);

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

            // Take top row of corners
            var refTop = refCorners.Take(checkerboardSize.Width).ToArray();
            var testTop = testCorners.Take(checkerboardSize.Width).ToArray();

            // Calculate vectors from first and last corner
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

            // Calculate angles
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
            // Number of pixels to check from each edge
            int borderSize = 20;
            var borders = new Dictionary<string, double>();

            // Get edge regions and calculate mean
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
}