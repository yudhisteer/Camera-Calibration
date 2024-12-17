import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraAlignmentChecker:
    def __init__(self, 
                 checkerboard_size=(7,7),  # Interior corners
                 square_size=100,          # Pixels
                 max_position_error=50,     # Pixels
                 max_rotation_error=5.0):   # Degrees
        """
        Initialize the camera alignment checker
        
        Args:
            checkerboard_size: Tuple of (rows, cols) of interior corners
            square_size: Size of each square in pixels
            max_position_error: Maximum allowed position offset in pixels
            max_rotation_error: Maximum allowed rotation in degrees
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.max_position_error = max_position_error
        self.max_rotation_error = max_rotation_error
        
    def generate_checkerboard(self):
        """Generate a checkerboard pattern"""
        rows, cols = self.checkerboard_size
        board_rows = rows + 1
        board_cols = cols + 1
        
        # Create the checkerboard image
        img_size = (self.square_size * board_rows, self.square_size * board_cols)
        img = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        
        # Fill in the squares
        for i in range(board_rows):
            for j in range(board_cols):
                if (i + j) % 2 == 0:
                    x1 = j * self.square_size
                    y1 = i * self.square_size
                    x2 = (j + 1) * self.square_size
                    y2 = (i + 1) * self.square_size
                    img[y1:y2, x1:x2] = 255
                    
        return img
    
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
    
    def calculate_alignment(self, reference_image, test_image):
        """Calculate alignment between reference and test images"""
        # Find corners in both images
        ref_corners = self.find_corners(reference_image)
        test_corners = self.find_corners(test_image)
        
        # Calculate center points
        ref_center = np.mean(ref_corners, axis=0)[0]
        test_center = np.mean(test_corners, axis=0)[0]
        
        # Calculate position offset
        position_error = np.linalg.norm(ref_center - test_center)
        
        # Calculate rotation
        # Using the top row of corners for rotation calculation
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
        
        return {
            'position_error': position_error,
            'rotation_error': rotation_error,
            'is_aligned': is_aligned
        }
    
    def check_alignment(self, test_image_path):
        """Main method to check camera alignment"""
        # Generate reference checkerboard
        ref_image = self.generate_checkerboard()
        
        # Save reference checkerboard
        cv2.imwrite('checkboard.png', ref_image)
        
        # Calculate alignment
        results = self.calculate_alignment(ref_image, test_image_path)
        
        # Print results
        print(f"Position Error: {results['position_error']:.2f} pixels")
        print(f"Rotation Error: {results['rotation_error']:.2f} degrees")
        print(f"Alignment Status: {'PASS' if results['is_aligned'] else 'FAIL'}")
        
        return results

# Usage example
if __name__ == "__main__":
    # Create checker with custom thresholds
    checker = CameraAlignmentChecker(
        checkerboard_size=(7,7),
        square_size=100,
        max_position_error=50,  # 50 pixels
        max_rotation_error=5.0  # 5 degrees
    )
    
    # Check alignment
    results = checker.check_alignment('test_image.png')