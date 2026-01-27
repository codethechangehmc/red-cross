"""
Building Detection using Pre-trained DINO Model
This script uses a pre-trained DINO model to identify buildings in images.
"""

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class BuildingDetector:
    def __init__(self):
        """Initialize the DINO model for building detection."""
        print("Loading pre-trained DINO model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Using Grounding DINO - a powerful zero-shot object detection model
        model_id = "IDEA-Research/grounding-dino-tiny"
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def detect_buildings(self, image_path, text_prompt="building", threshold=0.3):
        """
        Detect buildings in an image.
        
        Args:
            image_path: Path to the image file or URL
            text_prompt: Text description of what to detect (default: "building")
            threshold: Confidence threshold for detections (default: 0.3)
        
        Returns:
            Dictionary containing boxes, scores, and labels
        """
        # Load image
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[image.size[::-1]]
        )
        
        return results[0], image
    
    def visualize_results(self, image, results, save_path=None):
        """
        Visualize detection results on the image.
        
        Args:
            image: PIL Image
            results: Detection results from detect_buildings()
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels']
        
        print(f"\nDetected {len(boxes)} building(s):")
        
        # Draw bounding boxes
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{label}: {score:.2f}"
            ax.text(x1, y1 - 5, label_text,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                   fontsize=10, color='white')
            
            print(f"  {idx+1}. {label} - Confidence: {score:.3f} - Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"\nVisualization saved to: {save_path}")
        
        plt.show()
    
    def detect_and_visualize(self, image_path, text_prompt="building", threshold=0.3, save_path=None):
        """
        Convenience method to detect and visualize in one step.
        
        Args:
            image_path: Path to the image file or URL
            text_prompt: Text description of what to detect
            threshold: Confidence threshold for detections
            save_path: Optional path to save the visualization
        """
        print(f"Detecting '{text_prompt}' in image: {image_path}")
        results, image = self.detect_buildings(image_path, text_prompt, threshold)
        self.visualize_results(image, results, save_path)
        return results


def main():
    """Example usage of the BuildingDetector."""
    
    # Initialize detector
    detector = BuildingDetector()
    
    # Example: Detect buildings from a URL
    print("\n" + "="*70)
    print("Example: Detecting buildings from a sample image")
    print("="*70)
    
    sample_image = "https://images.unsplash.com/photo-1486718448742-163732cd1544?w=800"
    
    results = detector.detect_and_visualize(
        image_path=sample_image,
        text_prompt="building . house . skyscraper . architecture",
        threshold=0.25,
        save_path="sample_detection.jpg"
    )
    
    print("\n" + "="*70)
    print("Detection complete! Result saved to 'sample_detection.jpg'")
    print("\nTo process multiple images, run: python test_images.py")
    print("="*70)


if __name__ == "__main__":
    main()
