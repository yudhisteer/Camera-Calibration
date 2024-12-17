import cv2
import numpy as np

class AlignmentChecker:
    def __init__(self, checkerboard_size=(7,7), max_position_error=50, max_rotation_error=5.0):
        """
        Initialize the alignment checker
        
        Args:
            checkerboard_size: Tuple of (rows, cols) of interior corners
            max_position_error: Maximum allowed position offset in pixels
            max_rotation_error: Maximum allowed rotation in degrees
        """
        self.checkerboard_size = checkerboard_size
        self.max_position_error = max_position_error
        self.max_rotation_error = max_rotation_error
        
    def find_corners(self, image):
        """Find checkerboard corners in an image"""
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        # Find the corners
        ret, corners = cv2.findChessboardCorners(image, self.checkerboard_size, None)
        
        if not ret:
            raise ValueError("Could not find checkerboard corners in image")
            
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        
        return corners
    
    def check_alignment(self, reference_image_path, test_image_path):
        """Calculate alignment between reference and test images"""
        # Find corners in both images
        ref_corners = self.find_corners(reference_image_path)
        test_corners = self.find_corners(test_image_path)
        
        # Calculate center points
        ref_center = np.mean(ref_corners, axis=0)[0]
        test_center = np.mean(test_corners, axis=0)[0]
        
        # Calculate position offset
        position_error = np.linalg.norm(ref_center - test_center)
        
        # Calculate rotation
        ref_top = ref_corners[0:self.checkerboard_size[1]]
        test_top = test_corners[0:self.checkerboard_size[1]]
        
        ref_vector = ref_top[-1] - ref_top[0]
        test_vector = test_top[-1] - test_top[0]
        
        angle = np.arctan2(ref_vector[0][1], ref_vector[0][0]) - \
                np.arctan2(test_vector[0][1], test_vector[0][0])
        rotation_error = np.abs(np.degrees(angle))
        
        # Check against thresholds
        is_aligned = (position_error <= self.max_position_error and 
                     rotation_error <= self.max_rotation_error)
        
        results = {
            'position_error': position_error,
            'rotation_error': rotation_error,
            'is_aligned': is_aligned
        }
        
        # Print results
        print(f"Position Error: {results['position_error']:.2f} pixels")
        print(f"Rotation Error: {results['rotation_error']:.2f} degrees")
        print(f"Alignment Status: {'PASS' if results['is_aligned'] else 'FAIL'}")
        
        return results

if __name__ == "__main__":
    # Example usage
    checker = AlignmentChecker(
        checkerboard_size=(7,7),
        max_position_error=50,  # 50 pixels
        max_rotation_error=5.0  # 5 degrees
    )
    
    results = checker.check_alignment('checkboard.png', 'test_image.png')