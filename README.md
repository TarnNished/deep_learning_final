# Image Captioning with Attention

A deep learning project that generates descriptive captions for images using an encoder-decoder architecture with attention mechanism. The model uses ResNet18 as the encoder and an LSTM-based decoder with attention to generate natural language descriptions of images.

**Note:** This project is designed to run on Google Colab with data and models stored in Google Drive.

## Project Overview

This project implements an image captioning system that:
- Uses a pre-trained ResNet18 CNN as the encoder to extract visual features
- Employs an LSTM decoder with attention mechanism to generate captions
- Uses beam search for better caption generation during inference
- Includes a trained model ready for inference

## Architecture

- **Encoder**: ResNet18 (pre-trained) with spatial feature extraction (7×7×512 feature maps)
- **Attention**: Additive attention mechanism to focus on relevant image regions
- **Decoder**: LSTM-based decoder with word embeddings
- **Beam Search**: 3-beam search for caption generation

## Project Structure

```
deep_learning_final/
├── README.md
├── requirements.txt         # Python dependencies
├── artifacts/
│   ├── config.json          # Model configuration
│   ├── model.pt            # Trained model weights
│   └── vocab.json          # Vocabulary mapping
└── training_and_inference/
    ├── data_and_training.ipynb    # Training notebook
    └── inference.ipynb            # Inference notebook
```

## Requirements

### Python Dependencies

```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
tqdm>=4.60.0
jupyter>=1.0.0
```

## Google Drive Setup (Required)

This project is designed to run on **Google Colab** with files stored in **Google Drive**. You need to set up the following folder structure in your Google Drive:

### Option 1: Using the Pre-trained Model (Recommended for Quick Start)

If you want to **skip training** and use the already trained model:

1. **Create the artifacts folder** in your Google Drive:
   ```
   /content/drive/MyDrive/artifacts/
   ```

2. **Upload the trained model files** from the `artifacts/` folder to this location:
   - `config.json`
   - `model.pt`
   - `vocab.json`

3. **Your Google Drive structure should look like:**
   ```
   MyDrive/
   └── artifacts/
       ├── config.json
       ├── model.pt
       └── vocab.json
   ```

4. **Add test images:**
   ```
   MyDrive/
   └── caption_data/
       └── Images/
           ├── test1.jpg
           ├── test2.jpg
           └── test3.jpg
   ```

### Option 2: Training from Scratch

If you want to **train your own model**:

1. **Prepare your dataset** in Google Drive:
   ```
   MyDrive/
   └── caption_data/
       ├── Images/
       │   ├── img001.jpg
       │   ├── img002.jpg
       │   └── ... (all your images)
       └── captions.txt
   ```

2. **Format of `captions.txt`:**
   ```
   image,caption
   img001.jpg,a dog playing in the park
   img002.jpg,a sunset over the ocean
   img003.jpg,children playing on a beach
   ```
   - First line is the header
   - Each subsequent line: `image_filename,caption_text`

## Usage

### Running Inference (Using Pre-trained Model)

1. **Open Google Colab:**
   - Click the "Open in Colab" badge at the top of `inference.ipynb`
   - Or manually upload `inference.ipynb` to Google Colab

2. **Enable GPU runtime** (recommended):
   - Runtime → Change runtime type → GPU

3. **Mount Google Drive:**
   - Run the cell with `drive.mount('/content/drive')`
   - Follow the authentication prompts

4. **Verify your folder structure:**
   - The notebook expects files at `/content/drive/MyDrive/artifacts/`
   - Make sure you have uploaded `config.json`, `model.pt`, and `vocab.json` there

5. **Run all cells:**
   - The notebook will load the model and generate captions for test images
   - If you have test images at `/content/drive/MyDrive/caption_data/Images/`, they will be captioned

6. **Use your own images:**
   - Upload images to the `caption_data/Images/` folder in Google Drive
   - Modify the `test_images` list in the notebook to include your image filenames

### Training the Model

1. **Open Google Colab:**
   - Click the "Open in Colab" badge at the top of `data_and_training.ipynb`
   - Or manually upload to Colab

2. **Enable GPU runtime:**
   - Runtime → Change runtime type → GPU (essential for training)

3. **Set up your dataset** in Google Drive as described in "Option 2" above

4. **Mount Google Drive and verify paths:**
   ```python
   DATA_ROOT = "/content/drive/MyDrive/caption_data"
   IMAGE_DIR = os.path.join(DATA_ROOT, "Images")
   CAPTIONS_FILE = os.path.join(DATA_ROOT, "captions.txt")
   ARTIFACTS_DIR = "/content/drive/MyDrive/visual-storyteller/artifacts"
   ```

5. **Run all cells** to:
   - Load and preprocess the dataset
   - Build vocabulary from captions
   - Train the encoder-decoder model
   - Save trained model to `ARTIFACTS_DIR`

6. **Training time:**
   - Depends on dataset size and epochs
   - Default: 20 epochs with batch size 32
   - Monitor training with progress bars

### Training Configuration

The model uses the following hyperparameters (configurable in `CONFIG` dict):
- Image size: 224×224
- Embedding dimension: 256
- Hidden dimension: 512
- Batch size: 32
- Learning rate: 0.001
- Epochs: 20
- Max caption length: 30 tokens
- Minimum word frequency: 1

## Model Configuration

The `config.json` file contains:
```json
{
  "image_size": 224,
  "embedding_dim": 256,
  "hidden_dim": 512,
  "num_layers": 1,
  "batch_size": 32,
  "lr": 0.001,
  "epochs": 20,
  "max_len": 30,
  "min_word_freq": 1
}
```

## Features

- ✅ Attention mechanism for interpretable caption generation
- ✅ Beam search decoding for better quality captions
- ✅ Pre-trained ResNet18 encoder
- ✅ Repetition penalty to avoid duplicate words
- ✅ Special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`
- ✅ Efficient data loading with PyTorch DataLoader
- ✅ GPU acceleration support
- ✅ Ready-to-use trained model for inference
- ✅ Designed for Google Colab

## Quick Start Guide

### Just Want to Test the Model?

1. Download the `artifacts/` folder from this repository
2. Upload `config.json`, `model.pt`, and `vocab.json` to `/content/drive/MyDrive/artifacts/` in Google Drive
3. Open `inference.ipynb` in Google Colab
4. Run all cells
5. Upload your own images and generate captions!

### Want to Train Your Own Model?

1. Prepare your image dataset and captions.txt file
2. Upload to `/content/drive/MyDrive/caption_data/` in Google Drive
3. Open `data_and_training.ipynb` in Google Colab
4. Run all cells to train
5. Use the trained model with `inference.ipynb`

## Troubleshooting

### Common Issues:

1. **Google Drive mounting issues**
   - Run the `drive.mount('/content/drive')` cell
   - Follow the authentication link and paste the code
   - Ensure you're logged into the correct Google account

2. **FileNotFoundError: artifacts or model files not found**
   - Verify folder structure in Google Drive matches the paths in the notebook
   - Check that files are in `/content/drive/MyDrive/artifacts/`
   - Make sure you've uploaded `config.json`, `model.pt`, and `vocab.json`

3. **CUDA out of memory**
   - In Colab: Runtime → Factory reset runtime
   - Reduce batch size in CONFIG
   - Use a smaller image size

4. **Module not found errors**
   - Colab has most packages pre-installed
   - If needed, add `!pip install <package>` cells at the top of the notebook

5. **Path errors in notebooks**
   - Double-check all path variables match your Google Drive structure
   - Use absolute paths: `/content/drive/MyDrive/...`

6. **Dataset loading errors**
   - Ensure `captions.txt` has the correct format (header row, then `image,caption` per line)
   - Check that image files exist in the Images folder
   - Verify image filenames in captions.txt match actual files

## Example Workflow

```python
# After running all cells in inference.ipynb

# Generate caption for a single image
caption = generate_caption("/content/drive/MyDrive/caption_data/Images/myimage.jpg", model)
print(caption)

# The notebook also includes visualization
# It will display images with their generated captions
```

## Local Installation (Optional - Not Recommended)

If you prefer to run locally instead of Colab:

1. **Clone the repository**
   ```bash
   git clone https://github.com/TarnNished/deep_learning_final.git
   cd deep_learning_final
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Update paths in notebooks** to point to local directories instead of Google Drive paths

5. **Run Jupyter**
   ```bash
   jupyter notebook
   ```

**Note:** Training locally requires a CUDA-capable GPU. CPU training will be extremely slow.

## License

This project is for educational purposes.

## Acknowledgments

- ResNet architecture from torchvision.models
- Attention mechanism inspired by "Show, Attend and Tell" paper
- Dataset format follows standard image captioning datasets (e.g., Flickr8k, COCO)

## Contact

For questions or issues, please open an issue in the repository.
