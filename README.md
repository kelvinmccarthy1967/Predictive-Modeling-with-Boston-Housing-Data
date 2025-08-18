# Indoor Scenes Images Classification — CNNs & Transfer Learning (IndoorCVPR_09)

## Short Description
This project trains and evaluates image classification models on the **Indoor Scenes (indoorCVPR_09)** dataset. It compares three approaches:

1. **Baseline CNN** (from scratch)  
2. **CNN with data augmentation + dropout**  
3. **Transfer learning using MobileNetV3Large** (pretrained on ImageNet)  

The notebook/code performs:
- Dataset download (via Kaggle)  
- Preprocessing and dataset inspection  
- Model training and evaluation  

> The best performing approach in this run was the transfer-learning model, achieving ~61% validation accuracy.

---

## Key Facts & Quick Results

- **Dataset:** `itsahmad/indoor-scenes-cvpr-2019` (downloaded via kagglehub)  
  - Images placed under `indoorCVPR_09/Images`  
- **Number of images (after removing a few folders):** 14,056  
- **Number of classes:** 61  
- **Train / Validation split:** 80% / 20% → 11,245 training images, 2,811 validation images  
- **Image size:** 256 x 256 RGB  
- **Batch size:** 32  
- **Random seed:** 1001  

**Final Validation Accuracies Observed:**

| Model | Validation Accuracy |
|-------|------------------|
| Baseline CNN (from scratch) | ~20.10% |
| CNN + Augmentation + Dropout | ~24.37% |
| Transfer Learning (MobileNetV3Large, frozen backbone + top layers) | ~61.05% |

---

## Observations & Notes

- The **baseline model** quickly overfits to training data and achieves low validation accuracy (~20%).  
- Adding **augmentation and dropout** improves generalization slightly (~24%).  
- **Transfer learning** (frozen MobileNetV3Large backbone + small head) yields a substantially better result (~61% validation accuracy), likely because ImageNet-pretrained features generalize well to indoor scene recognition.  
- The dataset is fairly large (≈14k images) and multi-class (61 classes). Training will be faster with **GPU acceleration**.  

---

## Usage

1. Clone the repository:  
```bash
git clone <repo_url>

