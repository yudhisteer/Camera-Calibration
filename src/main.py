# main.py
import cv2
from checker import AlignmentChecker

def main():
    """Main function to run the alignment checker"""
    # Initialize the checker with desired parameters
    checker = AlignmentChecker(
        checkerboard_size=(7,7),
        max_rotation_error=5.0,
        max_scale_difference=0.05
    )
    
    # Run the alignment check
    try:
        results = checker.check_alignment('reference_screen.png', 'test_image.jpg')
        print("\nDetailed Results Dictionary:")
        print(results)
    except Exception as e:
        print(f"Error during alignment check: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()