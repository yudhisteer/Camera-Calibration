from generate_checkerboard import CheckerboardDisplay
from check_alignment import AlignmentChecker
import argparse

def main():
    parser = argparse.ArgumentParser(description='Camera Alignment Tool')
    parser.add_argument('--mode', choices=['generate', 'check', 'both'], 
                      help='Operation mode: generate checkerboard, check alignment, or both')
    parser.add_argument('--test-image', help='Path to test image (required for check and both modes)')
    args = parser.parse_args()
    
    if args.mode == 'generate' or args.mode == 'both':
        # Generate and display checkerboard
        display = CheckerboardDisplay(checkerboard_size=(7,7), square_size=100)
        display.display_fullscreen()
    
    if args.mode == 'check' or args.mode == 'both':
        if not args.test_image:
            print("Error: test image path is required for check mode")
            return
            
        # Check alignment
        checker = AlignmentChecker(
            checkerboard_size=(7,7),
            max_position_error=50,
            max_rotation_error=5.0
        )
        checker.check_alignment('checkboard.png', args.test_image)

if __name__ == "__main__":
    main()