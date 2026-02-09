# Hybrid Autism Detection System

This project is a hybrid, multimodal autism detection system designed to
analyze multiple behavioral and physiological indicators to assist in
early autism risk assessment.

## Project Overview
The system integrates three major components:
- Behavioral questionnaire analysis
- Speech signal processing using CNN-based models
- Eye-tracking simulation using MediaPipe

The outputs from these components are combined using a decision-level
fusion approach to provide a final prediction.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- CNN (Deep Learning)
- Streamlit
- SHAP (Explainability)

## Project Structure
autism_mainframe/
│── app.py
│── main.py
│── speech.py
│── gaze_utils.py
│── behavioral_model.ipynb
│── requirements.txt
│── shap_summary.png


## How It Works
1. User inputs behavioral responses and media inputs.
2. Separate models process behavioral, speech, and eye-tracking data.
3. Model outputs are combined using a hybrid fusion strategy.
4. Results are displayed through a simple dashboard interface.

## Important Note
Due to GitHub file size limitations, trained models (`.h5`, `.pkl`) and
large datasets (audio/video) are not included in this repository.
The code structure and preprocessing pipeline are provided for reference.

## Use Case
This project was developed as a real-world academic and research-oriented
system focusing on explainable and practical autism risk assessment rather
than clinical diagnosis.

## Author
**Medhini K S**  
MCA | Software Developer | 
