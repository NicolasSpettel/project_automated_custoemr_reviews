# Project NLP | Business Case: Automated Customer Reviews
### This project develops an automated system for analyzing customer reviews using Natural Language Processing (NLP). It tackles three main tasks: sentiment classification, product category clustering, and generative summarization, all accessible through a local web application.

## 1. Project Goal
The project's objective is to automate the analysis of product reviews by:

Classifying reviews as positive, negative, or neutral.

Clustering product categories into a few meta-categories.

Summarizing insights into a human-readable blog post format.

All components are integrated into a user-friendly Streamlit web application.

## 2. Setup and Installation
Prerequisites
Python: Version 3.8 or higher.

Conda: For environment management.

Google Colab: Recommended for GPU-heavy tasks.

OpenAI API Key: Required for the 2.5_sentiment_classification_comparison.ipynb notebook.

Hugging Face Account: To access the pre-trained models.

Environment Setup
Clone the repository:

Bash

git clone <your_repo_url>
cd <your_project_directory>
Create and activate the Conda environment:

Bash

conda create -n nlp_reviews python=3.8
conda activate nlp_reviews
Install dependencies from requirements.txt:

Bash

pip install -r requirements.txt
Note: Some packages required pip as they may not be available on conda.

## 3. Data and Models
Datasets
The project uses the Consumer Reviews of Amazon Products dataset from Kaggle.

Source: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data

The raw data files (Datafiniti_Amazon_Consumer_Reviews_of_FMCG_Products.csv, etc.) should be placed in the ./data/raw/ directory.

Models
Sentiment Classification: A fine-tuned cardiffnlp/twitter-roberta-base-sentiment-latest model is used. The model weights are hosted on Hugging Face and will be automatically downloaded by the code.

Hugging Face Hub: https://huggingface.co/Nicolas-Spettel/cardiffnlp_twitter-roberta-base-sentiment-latest

## 4. How to Run the Code
This section outlines the steps to reproduce the analysis and run the web application.

Step 1: Data Preparation and EDA
Run the 1_eda_and_preprocessing.ipynb notebook. This script handles data cleaning, merging the three CSV files, and initial exploration. It will generate processed CSV files in the ./data/processed/ directory.

Hardware: This step does not require a GPU.

Step 2: Sentiment Classification
Run 2_sentiment_classification_exploration_mlflow.ipynb to train and evaluate the sentiment model. This notebook uses MLflow for experiment tracking and hyperparameter tuning.

Hardware: Use Google Colab with a T4 GPU for faster training.

The trained model weights are saved to Hugging Face, as indicated above.

Step 3: Optional Model Comparison
Run 2.5_sentiment_classification_comparison.ipynb to compare your model's performance against the OpenAI API (GPT-3/4).

Hardware: This step requires an OpenAI API key in your .env file.

Step 4: Clustering Analysis
Run 3_clustering_analysis.ipynb. This notebook performs clustering on product categories to group similar products.

Hardware: Use Google Colab with a GPU for this task.

Step 5: Summarization Prototyping
Run 4_summarization_prototyping.ipynb. This notebook demonstrates how to use a generative model (e.g., T5, BART) to create blog-post-style summaries from the clustered reviews.

Hardware: This step does not require a heavy GPU.

Step 6: Running the Web Application
The Streamlit app serves as the main user interface.

Ensure all previous steps are completed and processed data files are in place.

Start the Streamlit server from the project root directory:

Bash

streamlit run app/app.py
Access the application at http://localhost:8501/ in your web browser.

The app provides a sidebar with dropdown options to explore each component of the project.

## 5. Project Structure
Of course. I will update the .gitignore to include image files and also adjust the directory structure to reflect the changes in the docs folder.

Here is the revised .gitignore file, now including common image file extensions to prevent them from being committed to the repository:

Code snippet

# Data and preprocessing
data/processed/
data/mlflow/
data/raw/

# Jupyter Notebook specifics
.ipynb_checkpoints/

# Python and environment files
__pycache__/
*.pyc
venv/
env/
conda.lock

# Machine Learning and Model-Related Files
mlruns/
*.db
*.pt
*.bin
*.safetensors

# IDE and OS-specific files
.DS_Store
Thumbs.db
.vscode/
.idea/

# Sensitive Information
.env

# Images and presentations
*.jpg
*.jpeg
*.png
*.gif
*.svg
*.pdf
The updated project structure reflecting the changes you mentioned would look like this:

5. Project Structure
├── app/
│   └── app.py
├── data/
│   ├── mlflow/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── final_model.txt
│   └── presentation_link.txt
├── notebooks/
│   ├── 1_eda_and_preprocessing.ipynb
│   ├── 2_sentiment_classification_exploration_mlflow.ipynb
│   ├── 2.5_sentiment_classification_comparison.ipynb
│   ├── 3_clustering_analysis.ipynb
│   └── 4_summarization_prototyping.ipynb
├── .env
├── .gitignore
├── pylintrc.txt
└── requirements.txt
