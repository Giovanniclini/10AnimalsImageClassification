
# 10 Animals Image Classification

This repository contains a project for image classification of 10 different animal classes, created during my exchange program at Hanyang University in Seoul, South Korea. The objective of this project was to build and train a convolutional neural network (CNN) to accurately classify images into their respective animal categories.

## Project Overview

The project involves:
- **Data Preprocessing**: Handling image data, normalization, augmentation, and cleaning.
- **Model Design**: Constructing a deep learning model using CNN.
- **Training**: Training the model with the dataset of 10 animal classes.
- **Evaluation**: Assessing model performance using accuracy and loss metrics.
- **Prediction**: Making predictions on new images.

## Dataset

The dataset used in this project is primarily sourced from the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) available on Kaggle. To enhance the diversity and quantity of the training data, additional images were collected from Pinterest using a web scraper. These images were then meticulously cleaned and curated to ensure consistency and quality within the dataset.

## Repository Structure

- `checkpoints/`: Saved model checkpoints.
- `test_images/`: Test images used to evaluate the model.
- `utils/`: Utility scripts for data processing and model management.
- `run.py`: Script for running the model.
- `train.py`: Script for training the model.
- `run_midterm.ipynb`: Jupyter notebook for model exploration and midterm results.
- `README.md`: Project documentation.

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Giovanniclini/10AnimalsImageClassification.git
   ```

2. Navigate to the repository directory:
   ```bash
   cd 10AnimalsImageClassification
   ```

3. Run the training notebook or experiment with the model to classify your own images.

## Results

The model achieves a high accuracy on the validation set, showcasing its ability to distinguish between the 10 animal classes effectively.

## Acknowledgements

This project was completed during my academic exchange at Hanyang University in Seoul, South Korea. 
If you need the full dataset just reach out!

---

Feel free to reach out if you have any questions or suggestions!
