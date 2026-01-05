## AlzhNet
AlzhNet is an end‑to‑end deep learning project for early detection of Alzheimer’s disease from brain MRI scans, trained on the popular multi‑class Kaggle dataset (NC / MCI / AD). The goal is not just to chase leaderboard numbers, but to build something a clinician could realistically use.

Under the hood, AlzhNet uses a custom CNN tailored for 2D MRI slices, trained with class‑balanced splits and data augmentation to handle the limited dataset size. During evaluation, the best checkpoint reaches around 79% test accuracy, with macro precision ≈ 0.82, macro recall ≈ 0.77, macro specificity ≈ 0.89, and macro ROC–AUC ≈ 0.93, showing that the model is not only accurate but also reasonably robust across all three classes, including the clinically critical MCI stage.

## Check the Project out here: https://alzhnet-alzheimers-mri-detection-4v3g8xbpbjc4rlfzerb7app.streamlit.app/​

# On top of the model, the repo includes a full pipeline:

DICOM‑aware data loader that mirrors a real MRI workflow (no manual PNG conversion).

Modular training code with clear logging, learning‑rate scheduling, and saved checkpoints.

Automatically generated training curves and confusion matrix for quick experiment review.

A Streamlit web app where you can upload an MRI slice and get a predicted class with confidence, making it easy to demo the system as a decision‑support tool rather than just a notebook experiment.
​
