Plant Disease Detection Using CNNs
This project aims to build a convolutional neural network (CNN) model for detecting plant diseases using image data. By training on a comprehensive dataset of plant images, the model can classify diseases with high accuracy, providing a valuable tool for agricultural health monitoring.

Dataset
The dataset used for this project can be found on Kaggle: New Plant Diseases Dataset.
The dataset includes images of leaves affected by various plant diseases, allowing for effective model training and testing.

Project Structure
data/: Contains scripts to load, process, and split the dataset.
notebooks/: Jupyter notebooks for data exploration, preprocessing, model building, and evaluation.
src/: Source code for the CNN model and related functions.
README.md: Project documentation.
Getting Started
Prerequisites
Python 3.7+

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Download Dataset
Download and extract the dataset from the Kaggle link and place it in the data/ directory.

Run Jupyter Notebook
Explore data and build models by opening notebooks/Plant_Disease_Detection.ipynb:

bash
Copy code
jupyter notebook notebooks/Plant_Disease_Detection.ipynb
Training the Model
Run src/train.py to train the model:

bash
Copy code
python src/train.py
Evaluating the Model
Run src/evaluate.py to evaluate the model’s accuracy and performance metrics:

bash
Copy code
python src/evaluate.py
Results
The trained model achieved high accuracy in classifying plant diseases. Results, including model accuracy and examples of predictions, can be found in notebooks/Model_Evaluation.ipynb.

Project Details
Model: Convolutional Neural Network (CNN)
Frameworks: TensorFlow, Keras
Evaluation Metrics: Accuracy, F1 Score, Precision, and Recall
Future Improvements
Hyperparameter tuning for improved accuracy.
Deployment as a web or mobile app for real-time disease detection in the field.
Expanding the dataset with additional plant species and disease types.
Contributing
If you’d like to contribute, feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT License.
