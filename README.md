# Building Detection with Pre-trained DINO Model

This project uses a pre-trained DINO (Grounding DINO) model to automatically identify and localize buildings in images.

## 📁 Project Structure

```
ARC1130/
├── building_detector.py      # Main detection class
├── test_images.py            # Batch processing script
├── requirements.txt          # Python dependencies
├── test_images/             # Input images (10 street view samples included)
└── detection_results/       # Output with annotated images (auto-created)
```

## ✨ Features

- **Pre-trained Model** - No training required, ready to use
- **Zero-shot Detection** - Detects buildings from text descriptions
- **Batch Processing** - Process multiple images automatically
- **Visual Output** - Draws bounding boxes with confidence scores
- **Flexible** - Works with local images or URLs

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Detection

**Option A: Test with sample online image**
```bash
python building_detector.py
```

**Option B: Process all test images**
```bash
python test_images.py
```
Processes all images in `test_images/` and saves results to `detection_results/`

## 📝 Usage Examples

### Basic Usage
```python
from building_detector import BuildingDetector

detector = BuildingDetector()

# Detect and visualize
results = detector.detect_and_visualize(
    image_path="test_images/your_image.jpg",
    text_prompt="building . house . skyscraper",
    threshold=0.3,
    save_path="detection_results/output.jpg"
)
```

### Advanced Usage
```python
# Get raw detection results
results, image = detector.detect_buildings(
    image_path="image.jpg",
    text_prompt="building",
    threshold=0.25
)

# Access data
boxes = results['boxes']      # Bounding box coordinates
scores = results['scores']    # Confidence scores (0-1)
labels = results['labels']    # Object labels
```

## ⚙️ Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `image_path` | Image file path or URL | `"test_images/img.jpg"` |
| `text_prompt` | Objects to detect (separate with `.`) | `"building . house . skyscraper"` |
| `threshold` | Confidence threshold (0.0-1.0) | `0.25` (recommended: 0.25-0.35) |
| `save_path` | Output file path (optional) | `"output.jpg"` |

### Text Prompt Examples
- `"building"` - General buildings
- `"building . house . skyscraper"` - Multiple types
- `"residential building . commercial building"` - By category
- `"modern building . glass building"` - By style

### Threshold Guide
- `0.1-0.2`: Very permissive (many detections, more false positives)
- `0.25-0.35`: **Balanced (recommended)**
- `0.4-0.6`: Conservative (fewer but more accurate)

## 🎯 Expected Results

**Example output:**
```
Detected 3 building(s):
  1. building - Confidence: 0.854 - Box: [120.5, 45.2, 780.3, 920.1]
  2. skyscraper - Confidence: 0.732 - Box: [800.1, 100.5, 1200.8, 950.3]
  3. house - Confidence: 0.621 - Box: [50.2, 600.3, 300.7, 850.9]
```

**Performance:**
- Successfully detects 9-19 buildings per image
- Confidence scores: 25%-59%
- Processing time: ~10-20 sec/image (CPU), ~2-5 sec/image (GPU)

## 🛠️ Technical Details

**Model:** Grounding DINO (IDEA-Research/grounding-dino-tiny)
- Type: Zero-shot object detection
- Size: ~600MB (downloads on first run)
- Device: Auto-detects GPU, falls back to CPU

**Dependencies:**
- Python 3.7+
- PyTorch 2.0+
- Transformers 4.35+
- Pillow, Matplotlib

## 🔧 Troubleshooting

**"No module named 'transformers'"**
```bash
pip install -r requirements.txt
```

**"No images found"**
- Ensure images are in `test_images/` folder
- Verify files are `.jpg` format

**Slow processing**
- Normal for CPU: 10-20 seconds per image
- Use GPU for 4-5x speedup

## 📊 Customization

To modify detection in `test_images.py`:
```python
results = detector.detect_and_visualize(
    image_path=image_path,
    text_prompt="building . house . skyscraper . facade",  # Customize
    threshold=0.25,                                        # Adjust
    save_path=output_path
)
```

## 📄 Citation

If using this code, please cite:
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [HuggingFace Transformers](https://huggingface.co/transformers)

---

**Last Updated:** December 2, 2025 | **Status:** Ready for deployment
