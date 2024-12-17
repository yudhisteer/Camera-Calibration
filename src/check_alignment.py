

import cv2
import numpy as np
import matplotlib.pyplot as plt

class AlignmentChecker:
    def __init__(self, checkerboard_size=(7,7), max_rotation_error=5.0, max_scale_difference=0.1):
        self.checkerboard_size = checkerboard_size
        self.max_rotation_error = max_rotation_error
        self.max_scale_difference = max_scale_difference
        plt.rcParams['figure.figsize'] = [12, 8]


    def draw_corners(self, image, corners, title="Corners"):
        """Draw detected corners on the image"""
        # Convert grayscale to RGB for colored visualization
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        # Determine orientation and ensure consistent first/last corner selection
        original_corners = corners.copy()
        top_row = corners[0:self.checkerboard_size[1]]
        first_point = top_row[0][0]
        last_point = top_row[-1][0]
        
        # Calculate if the pattern is more vertical or horizontal
        dx = abs(last_point[0] - first_point[0])
        dy = abs(last_point[1] - first_point[1])
        
        # If pattern is more vertical than horizontal, rotate the corner selection
        if dy > dx:
            # Rearrange corners to force horizontal interpretation
            new_corners = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 1, 2)
            new_corners = new_corners.transpose(1, 0, 2, 3)
            corners = new_corners.reshape(-1, 1, 2)
            
            # Check if we need to flip the order
            top_row = corners[0:self.checkerboard_size[0]]
            first_point = top_row[0][0]
            last_point = top_row[-1][0]
            if first_point[0] > last_point[0]:  # If first point is on the right
                # Flip the corners horizontally
                new_corners = corners.reshape(self.checkerboard_size[1], self.checkerboard_size[0], 1, 2)
                new_corners = np.flip(new_corners, axis=1)
                corners = new_corners.reshape(-1, 1, 2)
        else:
            # Check if we need to flip the order for horizontal pattern
            if first_point[0] > last_point[0]:  # If first point is on the right
                # Flip the corners horizontally
                new_corners = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 1, 2)
                new_corners = np.flip(new_corners, axis=1)
                corners = new_corners.reshape(-1, 1, 2)
        
        # Draw the checkerboard pattern with the corrected corners
        cv2.drawChessboardCorners(vis_img, self.checkerboard_size, corners, True)
        
        # Get the corrected top row for marker placement
        top_row = corners[0:self.checkerboard_size[1]]
        first_point = top_row[0][0]
        last_point = top_row[-1][0]
        
        # Draw first corner in blue
        first_corner = tuple(map(int, first_point))
        cv2.circle(vis_img, first_corner, 10, (0, 0, 255), 2)  # BGR to RGB
        cv2.putText(vis_img, "First", (first_corner[0]-20, first_corner[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw last corner in green
        last_corner = tuple(map(int, last_point))
        cv2.circle(vis_img, last_corner, 10, (0, 255, 0), 2)
        cv2.putText(vis_img, "Last", (last_corner[0]-20, last_corner[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw center point
        center = np.mean(corners, axis=0)[0]
        center = tuple(map(int, center))
        cv2.circle(vis_img, center, 10, (255, 255, 0), 2)  # BGR to RGB
        cv2.putText(vis_img, "Center", (center[0]-20, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw horizontal connections between corners
        corners_grid = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 2)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        
        for row in range(self.checkerboard_size[0]):
            for col in range(self.checkerboard_size[1]-1):
                start_point = tuple(map(int, corners_grid[row, col]))
                end_point = tuple(map(int, corners_grid[row, col+1]))
                color = colors[row % len(colors)]
                cv2.line(vis_img, start_point, end_point, color, 2)
        
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.title(title)
        plt.axis('on')
        plt.show()
        
        return vis_img


    def draw_bounds(self, image, corners, metrics, title="Bounds"):
        """Draw pattern bounds and measurements"""
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        # Get bounds
        min_x = int(np.min(corners[:,:,0]))
        max_x = int(np.max(corners[:,:,0]))
        min_y = int(np.min(corners[:,:,1]))
        max_y = int(np.max(corners[:,:,1]))
        
        # Draw bounding box
        cv2.rectangle(vis_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        # Draw center lines
        corners_grid = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 2)
        center_col = self.checkerboard_size[1] // 2
        center_row = self.checkerboard_size[0] // 2

        # Calculate vertical center line (in red)
        top_point = tuple(map(int, corners_grid[0, center_col]))
        bottom_point = tuple(map(int, corners_grid[-1, center_col]))
        cv2.line(vis_img, top_point, bottom_point, (0, 0, 255), 2)  # Red line

        # Calculate horizontal center line (in green)
        left_point = tuple(map(int, corners_grid[center_row, 0]))
        right_point = tuple(map(int, corners_grid[center_row, -1]))
        cv2.line(vis_img, left_point, right_point, (0, 0, 255), 2)  # Green line

        # Draw image center lines
        h, w = image.shape
        cv2.line(vis_img, (w//2, 0), (w//2, h), (255, 0, 0), 1)  # BGR to RGB
        cv2.line(vis_img, (0, h//2), (w, h//2), (255, 0, 0), 1)  # BGR to RGB
        
        # Add measurements text
        text_lines = [
            f"Pattern Size: {metrics['pattern_width']:.1f} x {metrics['pattern_height']:.1f}",
            f"Width Ratio: {metrics['width_ratio']:.3f}",
            f"Height Ratio: {metrics['height_ratio']:.3f}",
            f"Horizontal Ratio: {metrics['horizontal_ratio']:.3f}",
            f"Vertical Ratio: {metrics['vertical_ratio']:.3f}"
        ]
        
        for i, text in enumerate(text_lines):
            cv2.putText(vis_img, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.title(title)
        plt.axis('on')
        plt.show()
        
        return vis_img
        
    def find_corners(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
        if image is None:
            raise ValueError("Could not load image")
            
        self.image_height, self.image_width = image.shape
        
        # Show original image with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('on')
        plt.show()
        
        ret, corners = cv2.findChessboardCorners(image, self.checkerboard_size, None)
        
        if not ret:
            raise ValueError("Could not find checkerboard corners in image")
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        
        # Visualize detected corners
        vis_img = self.draw_corners(image, corners, "Detected Corners")
        
        return corners, image
    
    def calculate_pattern_metrics(self, corners, image):
        metrics = {
            'min_x': np.min(corners[:,:,0]), # Left
            'max_x': np.max(corners[:,:,0]), # Right
            'min_y': np.min(corners[:,:,1]), # Top
            'max_y': np.max(corners[:,:,1])  # Bottom
        }
        
        # Calculate distances from edges
        metrics['left_distance'] = metrics['min_x']
        metrics['right_distance'] = image.shape[1] - metrics['max_x']
        metrics['top_distance'] = metrics['min_y']
        metrics['bottom_distance'] = image.shape[0] - metrics['max_y']
        
        # Calculate pattern dimensions
        metrics['pattern_width'] = metrics['max_x'] - metrics['min_x']
        metrics['pattern_height'] = metrics['max_y'] - metrics['min_y']
        
        # Calculate ratios
        metrics['width_ratio'] = metrics['pattern_width'] / image.shape[1]
        metrics['height_ratio'] = metrics['pattern_height'] / image.shape[0]
        metrics['horizontal_ratio'] = metrics['left_distance'] / (metrics['left_distance'] + metrics['right_distance'])
        metrics['vertical_ratio'] = metrics['top_distance'] / (metrics['top_distance'] + metrics['bottom_distance'])
        
        # Add image dimensions
        metrics['image_width'] = image.shape[1]
        metrics['image_height'] = image.shape[0]
        
        # Visualize bounds and measurements
        vis_img = self.draw_bounds(image, corners, metrics, "Pattern Bounds")
        
        return metrics


    def check_alignment(self, reference_image_path, test_image_path):
        """Calculate alignment between reference and test images"""
        # Find corners in both images
        ref_corners, ref_image = self.find_corners(reference_image_path)
        test_corners, test_image = self.find_corners(test_image_path)

        # Log info
        #print(f"ref_corners: {ref_corners}, test_corners: {test_corners}")
        
        # Calculate metrics for both images
        ref_metrics = self.calculate_pattern_metrics(ref_corners, ref_image)
        test_metrics = self.calculate_pattern_metrics(test_corners, test_image)
        
        # Log info
        print(f"ref_metrics: {ref_metrics}, test_metrics: {test_metrics}")
        
        # Calculate differences
        horizontal_diff = abs(ref_metrics['horizontal_ratio'] - test_metrics['horizontal_ratio'])
        vertical_diff = abs(ref_metrics['vertical_ratio'] - test_metrics['vertical_ratio'])
        width_ratio_diff = abs(ref_metrics['width_ratio'] - test_metrics['width_ratio'])
        height_ratio_diff = abs(ref_metrics['height_ratio'] - test_metrics['height_ratio'])
        
        # Calculate rotation
        ref_top = ref_corners[0:self.checkerboard_size[1]] 
        test_top = test_corners[0:self.checkerboard_size[1]]

        # Log info
        print(f"ref_top: {ref_top}, test_top: {test_top}")


        #     First corner (0)          Last corner (-1)
        #      ↓                         ↓
        # •----•----•----•----•----•----•    
        # |    |    |    |    |    |    |
        # |    |    |    |    |    |    |
        
        # Vector = Last corner - First corner
        # →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→



        #     First corner       Last corner
        #      ↓                ↓
        # •----•----•----•----•----•----•
        #  \    \    \    \    \    \    \
        #   \    \    \    \    \    \    \
        
        # Vector shows the tilted angle
        # ↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗↗
        
        ref_vector = ref_top[-1] - ref_top[0]
        test_vector = test_top[-1] - test_top[0]

        # Log info
        print(f"ref_vector: {ref_vector}, test_vector: {test_vector}")
        
        angle = np.arctan2(ref_vector[0][1], ref_vector[0][0]) - \
                np.arctan2(test_vector[0][1], test_vector[0][0])
        rotation_error = np.abs(np.degrees(angle))
        
        # Define thresholds

        # Reference Image:
        # ┌──────────────────────┐
        # │      ░░░░░░░░        │
        # │   Checkerboard is    │
        # │    in the center     │
        # │      ░░░░░░░░        │
        # └──────────────────────┘
        # Left distance ≈ Right distance
        # Ratio ≈ 0.5 (centered)

        # Test Image (shifted right):
        # ┌──────────────────────┐
        # │            ░░░░░░░░  │
        # │         Checkerboard │
        # │         is shifted   │
        # │            ░░░░░░░░  │
        # └──────────────────────┘
        # Left distance > Right distance
        # Ratio ≈ 0.7 (not centered)

        max_position_ratio_diff = 0.1  # 10% difference in relative position
        
        # Check against thresholds
        is_horizontal_aligned = horizontal_diff <= max_position_ratio_diff
        is_vertical_aligned = vertical_diff <= max_position_ratio_diff
        is_rotation_aligned = rotation_error <= self.max_rotation_error
        is_scale_aligned = (width_ratio_diff <= self.max_scale_difference and 
                          height_ratio_diff <= self.max_scale_difference)
        
        is_aligned = (is_horizontal_aligned and is_vertical_aligned and is_rotation_aligned and is_scale_aligned)
        
        # Print detailed results
        print("\nAlignment Check Results:")
        print("\nPosition Analysis:")
        print(f"Horizontal alignment difference: {horizontal_diff:.3f} {'✓' if is_horizontal_aligned else '✗'}")
        print(f"Vertical alignment difference: {vertical_diff:.3f} {'✓' if is_vertical_aligned else '✗'}")
        
        print("\nScale Analysis:")
        print(f"Reference pattern/image width ratio: {ref_metrics['width_ratio']:.3f}")
        print(f"Test pattern/image width ratio: {test_metrics['width_ratio']:.3f}")
        print(f"Width ratio difference: {width_ratio_diff:.3f}")
        print(f"Height ratio difference: {height_ratio_diff:.3f}")
        print(f"Scale alignment: {'✓' if is_scale_aligned else '✗'}")
        
        print(f"\nRotation Error: {rotation_error:.2f}° {'✓' if is_rotation_aligned else '✗'}")
        print(f"\nOverall Status: {'PASS' if is_aligned else 'FAIL'}")
        
        return {
            'horizontal_difference': horizontal_diff,
            'vertical_difference': vertical_diff,
            'width_ratio_difference': width_ratio_diff,
            'height_ratio_difference': height_ratio_diff,
            'rotation_error': rotation_error,
            'is_horizontal_aligned': is_horizontal_aligned,
            'is_vertical_aligned': is_vertical_aligned,
            'is_rotation_aligned': is_rotation_aligned,
            'is_scale_aligned': is_scale_aligned,
            'is_aligned': is_aligned,
            'ref_metrics': ref_metrics,
            'test_metrics': test_metrics
        }


if __name__ == "__main__":
    checker = AlignmentChecker(
        checkerboard_size=(7,7),
        max_rotation_error=5.0,
        max_scale_difference=0.05
    )
    
    results = checker.check_alignment('reference_screen.png', 'test_image11.png')
    cv2.destroyAllWindows()