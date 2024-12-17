# checker.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from visualizer import AlignmentVisualizer

class AlignmentChecker:
    def __init__(self, checkerboard_size=(7,7), max_rotation_error=5.0, max_scale_difference=0.1):
        self.checkerboard_size = checkerboard_size
        self.max_rotation_error = max_rotation_error
        self.max_scale_difference = max_scale_difference
        self.visualizer = AlignmentVisualizer(checkerboard_size)

    def find_corners(self, image):
        """Find checkerboard corners in the image"""
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
        if image is None:
            raise ValueError("Could not load image")
            
        self.image_height, self.image_width = image.shape
        
        # Show original image
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('on')
        plt.show()
        
        # Find corners
        ret, corners = cv2.findChessboardCorners(image, self.checkerboard_size, None)
        if not ret:
            raise ValueError("Could not find checkerboard corners in image")
            
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        
        # Visualize detected corners
        self.visualizer.draw_corners(image, corners, "Detected Corners")
        
        return corners, image
    
    def calculate_pattern_metrics(self, corners, image):
        """Calculate pattern metrics from corners"""
        metrics = self._calculate_basic_metrics(corners, image)
        self.visualizer.draw_bounds(image, corners, metrics, "Pattern Bounds")
        return metrics

    def _calculate_basic_metrics(self, corners, image):
        """Calculate basic pattern metrics"""
        metrics = {
            'min_x': np.min(corners[:,:,0]),
            'max_x': np.max(corners[:,:,0]),
            'min_y': np.min(corners[:,:,1]),
            'max_y': np.max(corners[:,:,1])
        }
        
        # Calculate distances and dimensions
        metrics.update({
            'left_distance': metrics['min_x'],
            'right_distance': image.shape[1] - metrics['max_x'],
            'top_distance': metrics['min_y'],
            'bottom_distance': image.shape[0] - metrics['max_y'],
            'pattern_width': metrics['max_x'] - metrics['min_x'],
            'pattern_height': metrics['max_y'] - metrics['min_y'],
        })
        
        # Calculate ratios
        metrics.update({
            'width_ratio': metrics['pattern_width'] / image.shape[1],
            'height_ratio': metrics['pattern_height'] / image.shape[0],
            'horizontal_ratio': metrics['left_distance'] / (metrics['left_distance'] + metrics['right_distance']),
            'vertical_ratio': metrics['top_distance'] / (metrics['top_distance'] + metrics['bottom_distance']),
            'image_width': image.shape[1],
            'image_height': image.shape[0]
        })
        
        return metrics

    def check_alignment(self, reference_image_path, test_image_path):
        """Check alignment between reference and test images"""
        # Find corners in both images
        ref_corners, ref_image = self.find_corners(reference_image_path)
        test_corners, test_image = self.find_corners(test_image_path)
        
        # Calculate metrics
        ref_metrics = self.calculate_pattern_metrics(ref_corners, ref_image)
        test_metrics = self.calculate_pattern_metrics(test_corners, test_image)
        
        # Calculate differences
        differences = self._calculate_differences(ref_corners, test_corners, ref_metrics, test_metrics)
        
        # Check alignment
        alignment_status = self._check_alignment_status(differences)
        
        # Print results
        self._print_alignment_results(differences, alignment_status, ref_metrics, test_metrics)
        
        return {**differences, **alignment_status, 'ref_metrics': ref_metrics, 'test_metrics': test_metrics}

    def _calculate_differences(self, ref_corners, test_corners, ref_metrics, test_metrics):
        """Calculate differences between reference and test images"""
        horizontal_diff = abs(ref_metrics['horizontal_ratio'] - test_metrics['horizontal_ratio'])
        vertical_diff = abs(ref_metrics['vertical_ratio'] - test_metrics['vertical_ratio'])
        width_ratio_diff = abs(ref_metrics['width_ratio'] - test_metrics['width_ratio'])
        height_ratio_diff = abs(ref_metrics['height_ratio'] - test_metrics['height_ratio'])
        
        # Calculate rotation
        ref_top = ref_corners[0:self.checkerboard_size[1]]
        test_top = test_corners[0:self.checkerboard_size[1]]
        ref_vector = ref_top[-1] - ref_top[0]
        test_vector = test_top[-1] - test_top[0]
        
        angle = np.arctan2(ref_vector[0][1], ref_vector[0][0]) - \
                np.arctan2(test_vector[0][1], test_vector[0][0])
        rotation_error = np.abs(np.degrees(angle))
        
        return {
            'horizontal_difference': horizontal_diff,
            'vertical_difference': vertical_diff,
            'width_ratio_difference': width_ratio_diff,
            'height_ratio_difference': height_ratio_diff,
            'rotation_error': rotation_error
        }
    
    def _check_alignment_status(self, differences):
        """Check alignment status against thresholds"""
        max_position_ratio_diff = 0.1
        
        return {
            'is_horizontal_aligned': differences['horizontal_difference'] <= max_position_ratio_diff,
            'is_vertical_aligned': differences['vertical_difference'] <= max_position_ratio_diff,
            'is_rotation_aligned': differences['rotation_error'] <= self.max_rotation_error,
            'is_scale_aligned': (
                differences['width_ratio_difference'] <= self.max_scale_difference and 
                differences['height_ratio_difference'] <= self.max_scale_difference
            ),
        }

    def _print_alignment_results(self, differences, alignment_status, ref_metrics, test_metrics):
        """Print detailed alignment results"""
        print("\nAlignment Check Results:")
        print("\nPosition Analysis:")
        print(f"Horizontal alignment difference: {differences['horizontal_difference']:.3f} "
              f"{'✓' if alignment_status['is_horizontal_aligned'] else '✗'}")
        print(f"Vertical alignment difference: {differences['vertical_difference']:.3f} "
              f"{'✓' if alignment_status['is_vertical_aligned'] else '✗'}")
        
        print("\nScale Analysis:")
        print(f"Reference pattern/image width ratio: {ref_metrics['width_ratio']:.3f}")
        print(f"Test pattern/image width ratio: {test_metrics['width_ratio']:.3f}")
        print(f"Width ratio difference: {differences['width_ratio_difference']:.3f}")
        print(f"Height ratio difference: {differences['height_ratio_difference']:.3f}")
        print(f"Scale alignment: {'✓' if alignment_status['is_scale_aligned'] else '✗'}")
        
        print(f"\nRotation Error: {differences['rotation_error']:.2f}° "
              f"{'✓' if alignment_status['is_rotation_aligned'] else '✗'}")
        
        is_aligned = all(alignment_status.values())
        print(f"\nOverall Status: {'PASS' if is_aligned else 'FAIL'}")

