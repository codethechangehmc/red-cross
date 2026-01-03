# Data Collection & Preprocessing

This folder contains notebooks and scripts used to collect, organize, and document large-scale street-level imagery for the Red Cross disaster vulnerability mapping project.

## Overview
The primary goal of this stage is to build a reproducible data acquisition pipeline that gathers street-level and 360° imagery from the Mapillary API and structures it for downstream computer vision and geospatial analysis.

## Data Collection
- Collected **100,000+ images** using the Mapillary API
- Queried imagery for disaster-relevant geographic regions
- Handled API pagination, rate limits, and metadata retrieval
- Retrieved associated metadata including sequence IDs, timestamps, and camera information

## Data Organization
- Grouped images by **sequence ID** to preserve spatial and temporal continuity
- Organized imagery and metadata into structured directories for efficient access
- Prepared datasets suitable for future preprocessing, labeling, and model training

## Documentation & Reproducibility
- Implemented the pipeline in **Jupyter notebooks** with clear, step-by-step documentation
- Logged assumptions, API parameters, and intermediate outputs
- Designed notebooks to be reproducible and easy for collaborators to extend

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/codethechangehmc/red-cross.git
   cd red-cross/data_collection
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(If a requirements file is not present, install common dependencies such as `requests`, `pandas`, and `jupyter`.)*

4. Obtain a Mapillary API access token.

   Open the relevant data collection notebook and paste your access token into the designated variable cell (e.g., `ACCESS_TOKEN = "PASTE API KEY"`).

   *Note: The current implementation expects the token to be provided directly in the notebook. Environment variables are not required.*


5. Launch Jupyter/Google Colab/etc. and run the notebooks in order

   Follow the notebooks to collect imagery, retrieve metadata, and organize images by sequence.

## Next Steps

* Image preprocessing and filtering
* Integration with building footprint data
* Training computer vision models for building feature extraction

## Tools & Technologies

* Python
* Mapillary API
* Jupyter Notebooks
* Git/GitHub
