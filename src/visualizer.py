# visualizer.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

class AlignmentVisualizer:
    def __init__(self, checkerboard_size=(7,7)):
        self.checkerboard_size = checkerboard_size
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
            new_corners = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 1, 2)
            new_corners = new_corners.transpose(1, 0, 2, 3)
            corners = new_corners.reshape(-1, 1, 2)
            
            top_row = corners[0:self.checkerboard_size[0]]
            first_point = top_row[0][0]
            last_point = top_row[-1][0]
            if first_point[0] > last_point[0]:
                new_corners = corners.reshape(self.checkerboard_size[1], self.checkerboard_size[0], 1, 2)
                new_corners = np.flip(new_corners, axis=1)
                corners = new_corners.reshape(-1, 1, 2)
        else:
            if first_point[0] > last_point[0]:
                new_corners = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 1, 2)
                new_corners = np.flip(new_corners, axis=1)
                corners = new_corners.reshape(-1, 1, 2)
        
        # Draw the checkerboard pattern
        cv2.drawChessboardCorners(vis_img, self.checkerboard_size, corners, True)
        
        # Get the corrected top row for marker placement
        top_row = corners[0:self.checkerboard_size[1]]
        first_point = top_row[0][0]
        last_point = top_row[-1][0]
        
        # Draw markers
        self._draw_markers(vis_img, first_point, last_point, corners)
        
        # Draw horizontal connections
        self._draw_connections(vis_img, corners)
        
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.title(title)
        plt.axis('on')
        plt.show()
        
        return vis_img

    def _draw_markers(self, vis_img, first_point, last_point, corners):
        """Draw markers for first, last, and center points"""
        # First corner in blue
        first_corner = tuple(map(int, first_point))
        cv2.circle(vis_img, first_corner, 10, (0, 0, 255), 2)
        cv2.putText(vis_img, "First", (first_corner[0]-20, first_corner[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Last corner in green
        last_corner = tuple(map(int, last_point))
        cv2.circle(vis_img, last_corner, 10, (0, 255, 0), 2)
        cv2.putText(vis_img, "Last", (last_corner[0]-20, last_corner[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Center point
        center = np.mean(corners, axis=0)[0]
        center = tuple(map(int, center))
        cv2.circle(vis_img, center, 10, (255, 255, 0), 2)
        cv2.putText(vis_img, "Center", (center[0]-20, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def _draw_connections(self, vis_img, corners):
        """Draw horizontal connections between corners"""
        corners_grid = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 2)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        
        for row in range(self.checkerboard_size[0]):
            for col in range(self.checkerboard_size[1]-1):
                start_point = tuple(map(int, corners_grid[row, col]))
                end_point = tuple(map(int, corners_grid[row, col+1]))
                color = colors[row % len(colors)]
                cv2.line(vis_img, start_point, end_point, color, 2)

    def draw_bounds(self, image, corners, metrics, title="Bounds"):
        """Draw pattern bounds and measurements"""
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        # Draw bounds and center lines
        self._draw_bounding_box(vis_img, corners)
        self._draw_center_lines(vis_img, corners, image.shape)
        
        # Add measurements text
        self._draw_measurements(vis_img, metrics)
        
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.title(title)
        plt.axis('on')
        plt.show()
        
        return vis_img

    def _draw_bounding_box(self, vis_img, corners):
        """Draw bounding box around the pattern"""
        min_x = int(np.min(corners[:,:,0]))
        max_x = int(np.max(corners[:,:,0]))
        min_y = int(np.min(corners[:,:,1]))
        max_y = int(np.max(corners[:,:,1]))
        cv2.rectangle(vis_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    def _draw_center_lines(self, vis_img, corners, image_shape):
        """Draw center lines for pattern and image"""
        corners_grid = corners.reshape(self.checkerboard_size[0], self.checkerboard_size[1], 2)
        center_col = self.checkerboard_size[1] // 2
        center_row = self.checkerboard_size[0] // 2

        # Pattern center lines
        top_point = tuple(map(int, corners_grid[0, center_col]))
        bottom_point = tuple(map(int, corners_grid[-1, center_col]))
        cv2.line(vis_img, top_point, bottom_point, (0, 0, 255), 2)

        left_point = tuple(map(int, corners_grid[center_row, 0]))
        right_point = tuple(map(int, corners_grid[center_row, -1]))
        cv2.line(vis_img, left_point, right_point, (0, 0, 255), 2)

        # Image center lines
        h, w = image_shape
        cv2.line(vis_img, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        cv2.line(vis_img, (0, h//2), (w, h//2), (255, 0, 0), 1)

    def _draw_measurements(self, vis_img, metrics):
        """Draw measurement text on the image"""
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