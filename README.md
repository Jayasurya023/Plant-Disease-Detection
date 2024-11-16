# 🌿 Plant Disease Detection Using CNNs 🌿

This project leverages a **Convolutional Neural Network (CNN)** model for detecting plant diseases from leaf images. The goal is to build a highly accurate model to assist in identifying and managing plant health, which can be a valuable tool in agriculture. 


## 📂 Dataset

The dataset used for this project can be found on Kaggle: [🌱 New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).  
It includes images of leaves with various diseases, which are crucial for training and testing the model.

## 🗂️ Project Structure

- **📁 data/**: Scripts to load, process, and split the dataset.
- **📓 notebooks/**: Jupyter notebooks for data exploration, preprocessing, model building, and evaluation.
- **📜 src/**: Source code for the CNN model and related functions.
- **📘 README.md**: Project documentation.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### ⚙️ Usage

1. **📥 Download Dataset**  
   Download and extract the dataset from the [Kaggle link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and place it in the `data/` directory.

2. **📊 Run Jupyter Notebook**  
   Explore data and build models by opening `notebooks/Plant_Disease_Detection.ipynb`:

    ```bash
    jupyter notebook notebooks/Plant_Disease_Detection.ipynb
    ```

3. **🧠 Training the Model**  
   Run `src/train.py` to train the model:

    ```bash
    python src/train.py
    ```

4. **📈 Evaluating the Model**  
   Run `src/evaluate.py` to evaluate the model’s accuracy and performance metrics:

    ```bash
    python src/evaluate.py
    ```

### 🏆 Results

The trained model achieved high accuracy in classifying plant diseases. Results, including model accuracy and examples of predictions, can be found in `notebooks/Model_Evaluation.ipynb`.

## 🔍 Project Details

- **Model**: Convolutional Neural Network (CNN)
- **Frameworks**: TensorFlow, Keras
- **Evaluation Metrics**: Accuracy, F1 Score, Precision, and Recall

## 📈 Future Improvements

- 🔧 Hyperparameter tuning for improved accuracy.
- 🌐 Deployment as a web or mobile app for real-time disease detection in the field.
- 🌍 Expanding the dataset with additional plant species and disease types.

## 🤝 Contributing

If you’d like to contribute, feel free to fork the repository and submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

Feel free to add screenshots of the model results, performance graphs, or the UI if you're deploying a web app.
