# AI Medical Imaging Diagnosis using Deep Learning

- This project is a **medical image classification system** built with **TensorFlow/Keras**.  
- It uses a Convolutional Neural Network (CNN) trained on medical imaging datasets to classify images into categories.  
- The project supports both **local deployment (Flask)** and **cloud-based demo (Gradio in Google Colab)**.

---

## Project Structure
   ```bash
    Medical Imaging/
    │
    ├── app.py # Flask web app for local deployment
    ├── train_model.py # Script to train the CNN model
    ├── requirements.txt # Python dependencies
    ├── README.md # Project documentation
    │
    ├── data/ # Dataset directory
    │ ├── train/ # Training images (organized by class folders)
    │ ├── val/ # Validation images (organized by class folders)
    │ └── test/ # Testing images (organized by class folders)
    │
    └── model/ # Saved model directory
    └── medical_cnn.h5 # Trained CNN model


    ---
```

## Installation (Local)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medical-imaging-diagnosis.git
   cd medical-imaging-diagnosis


2. Create a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows


3. Install dependencies:
   ```bash
    pip install -r requirements.txt

## Training the Model

1. Place your dataset inside the data/ folder with train, val, and test subfolders.
Each class should be in its own subdirectory (e.g., data/train/class1/, data/train/class2/).

2. Run the training script:
   ```bash
    python train_model.py


3. The trained model will be saved as:
   ```bash
    model/medical_cnn.h5

## Running the Application
### Option 1: Local Flask App

1. Ensure you have a trained model (model/medical_cnn.h5).

2. Run the Flask app:
   ```bash
    python app.py


3. Open in browser:
   ```bash
    http://127.0.0.1:5000

### Option 2: Google Colab with Gradio (Recommended)

1. Upload your project files to Colab.

2. Install Gradio:
   ```bash
    !pip install gradio


3. Example Gradio demo code:
   ```bash
    import gradio as gr
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np

    # Load model
    model = tf.keras.models.load_model("model/medical_cnn.h5")

    # Prediction function
    def predict(img):
        img = img.resize((128,128))
        x = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = model.predict(x)
        return {"Class 0": float(pred[0][0]), "Class 1": float(pred[0][1])}

    # Launch interface
    demo = gr.Interface(fn=predict, inputs=gr.Image(), outputs="label")
    demo.launch()


4. A public URL will be generated to test predictions.

### Requirements

- Python 3.8+

- TensorFlow / Keras

- Flask (for local deployment)

- Gradio (for Colab demo)

- Numpy, Pandas, Matplotlib

Install with:
   ```bash
    pip install -r requirements.txt
```
### Notes

- Training should be performed in Google Colab with GPU for faster execution.

- For deployment, use Flask locally or integrate with Streamlit/Gradio for easier cloud-based demos.

- Dataset should follow the folder structure:
   ```bash
    data/
    ├── train/
    │   ├── class1/
    │   └── class2/
    ├── val/
    │   ├── class1/
    │   └── class2/
    └── test/
        ├── class1/
        └── class2/

## License

This project is for academic and research purposes only. Not intended for real-world medical diagnosis.
