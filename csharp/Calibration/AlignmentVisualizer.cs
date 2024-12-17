using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Linq;
using System.Windows.Forms;
using Emgu.CV.Util;

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
        // Convert grayscale to RGB for colored visualization
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
        // Convert PointF array to VectorOfPointF for EmguCV compatibility
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

        // Display image
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
        // Create a smaller display image
        Mat displayImage = new Mat();
        Size displaySize = new Size(800, 600);
        CvInvoke.Resize(image, displayImage, displaySize);

        // Show image in window
        CvInvoke.NamedWindow(title, WindowFlags.Normal);
        CvInvoke.Imshow(title, displayImage);
        CvInvoke.WaitKey(1); // Update the window

        // Save the original (non-resized) image
        string filename = title.Replace(" ", "_") + ".png";
        image.Save(filename);
        Console.WriteLine($"Saved image to {filename}");

        // Clean up
        displayImage.Dispose();
    }
}
