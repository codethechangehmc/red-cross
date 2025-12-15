"""
Test building detection on PNG images only
"""

from building_detector import BuildingDetector
import os
import glob

def main():
    # Initialize detector
    print("Initializing Building Detector...")
    detector = BuildingDetector()
    
    # Define directories
    input_dir = "test_images"
    output_dir = "detection_results"
    
    # Get only PNG files
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        print(f"No PNG images found in '{input_dir}/' directory!")
        return
    
    print(f"\nFound {len(image_files)} PNG images to process\n")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for idx, image_path in enumerate(sorted(image_files), 1):
        # Get just the filename for display
        filename = os.path.basename(image_path)
        print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        print("-"*70)
        
        # Generate output filename (keep as png)
        output_filename = f"detected_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Detect buildings (or non-buildings)
            results = detector.detect_and_visualize(
                image_path=image_path,
                text_prompt="building . house . skyscraper . architecture . facade",
                threshold=0.25,
                save_path=output_path
            )
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print(f"Processing complete! Results saved to '{output_dir}/' folder")
    print("="*70)


if __name__ == "__main__":
    main()
