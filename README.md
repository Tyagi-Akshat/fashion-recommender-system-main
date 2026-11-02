# 👗 Fashion Recommender System

A deep learning-based **Fashion Recommender System** that recommends visually similar clothing and fashion items using image similarity. The project leverages a pretrained **ResNet50 CNN** to extract image features and **K-Nearest Neighbors (KNN)** to identify similar products.

Users can upload an image of a fashion item through an interactive Streamlit application and receive recommendations from the existing image dataset.

## ✨ Features

- **Deep Learning Feature Extraction:** Uses pretrained ResNet50 to extract meaningful visual features from fashion images.
- **Transfer Learning:** Utilizes ImageNet pretrained weights without retraining the entire network.
- **Image Similarity Search:** Finds visually similar fashion items using the K-Nearest Neighbors algorithm.
- **Interactive Web Interface:** Provides a simple interface built with Streamlit for uploading images and viewing recommendations.
- **Efficient Feature Storage:** Stores precomputed image embeddings and filenames using Pickle to avoid repeated feature extraction during normal application use.
- **Visual Recommendations:** Displays similar fashion items based on the uploaded image.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning and feature extraction |
| ResNet50 | Pretrained CNN for image representations |
| NumPy | Numerical operations and feature processing |
| Scikit-learn | K-Nearest Neighbors similarity search |
| Streamlit | Interactive web application |
| Pillow (PIL) | Image loading and processing |
| Pickle | Storage and loading of extracted features |
| tqdm | Progress tracking during feature extraction |

## 🧠 How It Works

The recommendation pipeline consists of the following steps:

### 1. Fashion Image Dataset

The system uses a collection of fashion images stored in the `images/` directory. These images form the reference dataset against which uploaded images are compared.

### 2. Feature Extraction Using ResNet50

A pretrained ResNet50 model is loaded with ImageNet weights. Its final classification layer is excluded, and a Global Max Pooling layer is added to obtain feature representations.

Each image is:
- Resized to 224 × 224 pixels.
- Converted into a numerical array.
- Preprocessed according to ResNet50 requirements.
- Passed through the CNN to extract visual features.
- Flattened and normalized to produce a feature vector.

### 3. Feature Storage

The extracted feature vectors and corresponding image paths are stored in two files:

- `embeddings.pkl` — contains the precomputed image feature vectors.
- `filenames.pkl` — contains the paths of the corresponding fashion images.

This allows the application to load the existing feature database instead of extracting features from the entire dataset every time it starts.

### 4. Similarity-Based Recommendation

When a user uploads a fashion image, the system extracts its feature vector using the same ResNet50 pipeline.

The feature vector is compared with the stored dataset using Scikit-learn's `NearestNeighbors` implementation with:
- Algorithm: Brute force
- Distance metric: Euclidean distance
- Number of neighbors: 6, including the nearest match

The system then displays five recommended fashion images from the retrieved results.

### 5. Interactive Visualization

The Streamlit interface allows users to upload an image and view the uploaded item alongside visually similar products.

## 🏗️ Project Architecture

```text
Fashion Recommender System
│
├── images/
│   └── Fashion image dataset
│
├── uploads/
│   └── Uploaded user images
│
├── app.py
│   └── Extracts image features using ResNet50
│       and saves embeddings and filenames
│
├── main.py
│   └── Runs the Streamlit recommendation interface
│
├── embeddings.pkl
│   └── Precomputed image feature vectors
│
├── filenames.pkl
│   └── Paths to dataset images
│
├── test.py
│   └── Testing script
│
└── README.md
```

**Note:** The `images/` directory and the generated Pickle files must be available before running the recommendation interface. The `uploads/` directory must also exist for the current upload implementation.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.9 or a compatible Python version supported by your chosen TensorFlow release
- pip
- A fashion image dataset

### 1. Clone the Repository

```bash
git clone https://github.com/Tyagi-Akshat/fashion-recommender-system-main.git
```

Navigate to the project directory:

```bash
cd fashion-recommender-system-main
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow streamlit numpy scikit-learn pillow tqdm
```

### 4. Prepare the Dataset

Create the required directories if they do not already exist:

```bash
mkdir images uploads
```

Place your fashion images inside the `images/` directory.

For example:

```text
images/
├── shirt1.jpg
├── shirt2.jpg
├── shoes1.jpg
├── jacket1.jpg
└── dress1.jpg
```

These are illustrative filenames; use your actual dataset images.

### 5. Generate Image Embeddings

Run the feature extraction script:

```bash
python app.py
```

The script processes the images in the `images/` directory and generates:

```text
embeddings.pkl
filenames.pkl
```

The first execution may take some time, depending on the size of your dataset and available hardware.

### 6. Launch the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

Open the local URL displayed in your terminal, typically:

```text
http://localhost:8501
```

Upload a fashion image to see the recommended similar items.

## 📂 Main Files

| File | Description |
|---|---|
| `app.py` | Loads ResNet50, extracts features from the fashion image dataset, and saves the feature vectors and filenames. |
| `main.py` | Implements the Streamlit interface, processes uploaded images, finds nearest neighbors, and displays recommendations. |
| `embeddings.pkl` | Stores the precomputed feature vectors for the reference dataset. |
| `filenames.pkl` | Stores the image paths associated with the extracted features. |
| `test.py` | Contains the project's testing code. |

## ⚙️ Recommendation Algorithm

The project uses a content-based recommendation approach.

Rather than recommending products based on user ratings or purchase history, it compares the visual characteristics of images.

**Workflow:**

```text
Input Fashion Image
        |
        v
Image Preprocessing
        |
        v
ResNet50 Feature Extraction
        |
        v
Feature Vector Normalization
        |
        v
KNN Similarity Search
        |
        v
Retrieve Nearest Images
        |
        v
Display Fashion Recommendations
```

The Euclidean distance between feature vectors determines the nearest neighbors. Smaller distances indicate greater proximity in the extracted feature space.

## 🔍 Content-Based vs. Collaborative Filtering

This implementation uses content-based recommendations.

- **Content-based filtering:** Recommends items based on their visual similarity to the uploaded image.
- **Collaborative filtering:** Recommends items using interactions or preferences from multiple users.

The content-based approach is appropriate for this project because recommendations can be generated directly from an image without requiring a history of user ratings or purchases.

## 💡 Potential Applications

- Fashion discovery and visual product search
- Online clothing stores
- Similar-product recommendation systems
- Fashion catalog exploration
- Image-based shopping experiences

## 🔮 Future Improvements

Potential enhancements include:

- Adding fashion categories and metadata to improve recommendation relevance.
- Exploring approximate nearest-neighbor search for larger datasets.
- Improving the user interface with filters and category selection.
- Deploying the application to a cloud hosting platform.
- Adding product descriptions, prices, and shopping links.
- Evaluating recommendation quality using a labeled dataset and appropriate retrieval metrics.
- Improving upload validation and handling missing or invalid image files.

## ⚠️ Limitations

- Recommendation quality depends on the image dataset and the features extracted by ResNet50.
- Visual similarity does not necessarily imply identical product categories, styles, or prices.
- Brute-force nearest-neighbor search can become computationally expensive as the dataset grows.
- The application requires precomputed embeddings and image paths that correspond to the current dataset.
- The project currently focuses on image similarity rather than personalized recommendations based on individual user preferences.

## 👨‍💻 Author

**Akshat Tyagi**

- GitHub: [Tyagi-Akshat](https://github.com/Tyagi-Akshat)
- Project Repository: [Fashion Recommender System](https://github.com/Tyagi-Akshat/fashion-recommender-system-main)

## 📄 License

No license is currently specified in this README. Add a suitable open-source license if you intend to permit others to reuse, modify, or distribute the project.
