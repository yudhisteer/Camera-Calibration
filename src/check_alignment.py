import cv2
import numpy as np

class AlignmentChecker:
    def __init__(self, checkerboard_size=(7,7), max_rotation_error=5.0, 
                 max_center_offset=20, max_scale_difference=0.1):
        """
        Initialize the alignment checker
        
        Args:
            checkerboard_size: Tuple of (rows, cols) of interior corners
            max_rotation_error: Maximum allowed rotation in degrees
            max_center_offset: Maximum allowed offset from center in pixels
            max_scale_difference: Maximum allowed difference in relative size (0.1 = 10%)
        """
        self.checkerboard_size = checkerboard_size
        self.max_rotation_error = max_rotation_error
        self.max_center_offset = max_center_offset
        self.max_scale_difference = max_scale_difference
        
    def find_corners(self, image):
        """Find checkerboard corners in an image"""
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
        if image is None:
            raise ValueError("Could not load image")
            
        # Store image dimensions for center calculation
        self.image_height, self.image_width = image.shape
        
        # Find the corners
        ret, corners = cv2.findChessboardCorners(image, self.checkerboard_size, None)
        
        if not ret:
            raise ValueError("Could not find checkerboard corners in image")
            
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        
        return corners, image
    
    def calculate_pattern_metrics(self, corners, image):
        """Calculate relative position and scale metrics of the checkerboard"""
        # Calculate checkerboard bounds
        min_x = np.min(corners[:,:,0])
        max_x = np.max(corners[:,:,0])
        min_y = np.min(corners[:,:,1])
        max_y = np.max(corners[:,:,1])
        
        # Calculate distances from edges
        left_distance = min_x
        right_distance = image.shape[1] - max_x
        top_distance = min_y
        bottom_distance = image.shape[0] - max_y
        
        # Calculate pattern width and height
        pattern_width = max_x - min_x
        pattern_height = max_y - min_y
        
        # Calculate scale ratios (pattern size relative to image size)
        width_ratio = pattern_width / image.shape[1]
        height_ratio = pattern_height / image.shape[0]
        
        # Calculate relative position ratios
        horizontal_ratio = left_distance / (left_distance + right_distance)
        vertical_ratio = top_distance / (top_distance + bottom_distance)
        
        return {
            'horizontal_ratio': horizontal_ratio,
            'vertical_ratio': vertical_ratio,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'pattern_width': pattern_width,
            'pattern_height': pattern_height,
            'image_width': image.shape[1],
            'image_height': image.shape[0]
        }
    
    def check_alignment(self, reference_image_path, test_image_path):
        """Calculate alignment between reference and test images"""
        # Find corners in both images
        ref_corners, ref_image = self.find_corners(reference_image_path)
        test_corners, test_image = self.find_corners(test_image_path)
        
        # Calculate metrics for both images
        ref_metrics = self.calculate_pattern_metrics(ref_corners, ref_image)
        test_metrics = self.calculate_pattern_metrics(test_corners, test_image)
        
        # Calculate differences
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
        
        # Define thresholds
        max_position_ratio_diff = 0.1  # 10% difference in relative position
        
        # Check against thresholds
        is_horizontal_aligned = horizontal_diff <= max_position_ratio_diff
        is_vertical_aligned = vertical_diff <= max_position_ratio_diff
        is_rotation_aligned = rotation_error <= self.max_rotation_error
        is_scale_aligned = (width_ratio_diff <= self.max_scale_difference and 
                          height_ratio_diff <= self.max_scale_difference)
        
        is_aligned = (is_horizontal_aligned and is_vertical_aligned and 
                     is_rotation_aligned and is_scale_aligned)
        
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
    # Example usage
    checker = AlignmentChecker(
        checkerboard_size=(7,7),
        max_rotation_error=5.0,       # 5 degrees
        max_center_offset=20,         # 20 pixels
        max_scale_difference=0.06      # 10% size difference tolerance
    )
    
    results = checker.check_alignment('reference_screen.png', 'test_image11.png')