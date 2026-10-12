# Line-Curve Block DeblurGAN

A research-focused image restoration project for recovering motion-blurred and distorted Bangla vehicle license plate images using a modified DeblurGAN-v2 generator with a custom Line-Curve block.

The implementation is centered on preserving Bangla character geometry, including horizontal matras, vertical strokes, curves, and structural details that are often lost under motion or defocus blur. This repository documents the model architecture, dataset preparation, training setup, and experimental evaluation performed in the local project workspace.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Research Motivation](#research-motivation)
- [Proposed Methodology](#proposed-methodology)
- [Line-Curve Block Architecture](#line-curve-block-architecture)
- [Model Architecture and Technical Details](#model-architecture-and-technical-details)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Training](#training)
- [Inference and Usage](#inference-and-usage)
- [Experimental Results](#experimental-results)
- [Model Comparison](#model-comparison)
- [Repository Structure](#repository-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Limitations and Future Work](#limitations-and-future-work)
- [Research Context and References](#research-context-and-references)
- [Author and Acknowledgments](#author-and-acknowledgments)

## Project Overview

This project addresses a practical and challenging computer vision problem: restoring severely blurred Bangla vehicle license plate images so that the textual content remains readable and structurally faithful. The design is not a generic deblurring pipeline; it is tailored to preserve the unique visual shape of Bangla characters, which are especially sensitive to blur because they contain thin strokes, slanted curves, and dense combinations of connected glyph structures.

The repository contains a modified DeblurGAN-v2-style generator trained on paired blurred-clean license plate images. The main architectural contribution is a custom `LineCurveBlock` that explicitly models horizontal structure, vertical structure, and curve-like deformation using a combination of directional convolutions and deformable convolution. The block is inserted throughout the network to improve the restoration of fine character detail while keeping the computational structure close to a convolutional generator baseline.

The implementation is primarily notebook-based and is stored in the project root as `Line-Curve Block DelburGAN.ipynb`, which defines the dataset pipeline, generator/discriminator, loss functions, training loop, and evaluation logic.

**Blur terminology:** The project addresses defocus blur and motion blur. Some existing folder names, notebook variables, and saved CSV values still use `Gaussian`/`gaussian` as a legacy label for the defocus-blurred examples; this README uses **defocus blur** when describing the task and calls out the legacy label where it appears.

## Key Features

- Modified DeblurGAN-v2-style generator for license plate image restoration.
- Targets both defocus blur and motion blur. In existing artifact names, the defocus subset may be labeled `Gaussian` or `gaussian`.
- Custom `LineCurveBlock` with three parallel processing branches:
  - horizontal strip convolution
  - vertical strip convolution
  - deformable curve-aware convolution
- Residual-style fusion that combines directional features and learns a gated feature response.
- Edge-aware restoration objective based on Sobel gradient difference.
- Perceptual feature loss using a pre-trained VGG19 backbone.
- GAN-based adversarial objective for realistic restoration.
- Dataset split into training, validation, and test sets using a folder-based paired image setup.
- Evaluation metrics for PSNR, SSIM, and MAE computed on blurred and restored outputs.
- Local checkpointing for best and latest model states.
- Comparison with baseline restoration models saved in the project output folders.

## Research Motivation

Bangla license plates are difficult to restore because the characters are visually complex, densely arranged, and highly sensitive to blur. Horizontal matras, vertical components, and curved strokes are often merged or lost when motion blur or defocus blur degrades the image. Standard generic restoration networks can recover global contrast but may fail to preserve fine character geometry, which is critical for plate readability and downstream OCR.

This project focuses on a structural restoration objective: the model should not merely smooth the image, but should maintain the topology of Bangla characters. The custom `LineCurveBlock` is designed around that idea by explicitly separating feature extraction into directional and deformable subpaths, making the network more responsive to long repeated strokes and curved regions.

## Proposed Methodology

The workflow implemented in the project follows a paired-image supervised restoration pipeline:

1. Clean license plate images are paired with defocus-blurred and motion-blurred versions. The defocus subset is stored under legacy `gaussian` folder/variable names.
2. The blurred image is processed by the generator.
3. The generator uses downsampling and multi-scale residual feature extraction with `LineCurveBlock` modules.
4. A discriminator evaluates realism and adversarial consistency.
5. The network is trained using a combined objective involving L1 reconstruction, edge preservation, perceptual similarity, and adversarial loss.
6. The model is evaluated on validation/test images using PSNR, SSIM, and MAE.

![GAN overview showing the generator, PatchGAN discriminator, and training losses](figures/gan_overview_diagram.png)

## Line-Curve Block Architecture

The central contribution of the project is the custom `LineCurveBlock`, implemented in the main notebook. The block is designed to capture the two dominant structural directions in Bangla characters: horizontal strokes and vertical strokes, while also learning curvilinear patterns through a deformable path.

### Generator architecture

![Generator architecture with Line-Curve Blocks](figures/fig2_generator_architecture.png)

### Line-Curve Block diagram

![Line-Curve Block branches, fusion, and gating](figures/line_curve_block_diagram.png)

The actual implementation from the notebook shows the following structure:

```python
class LineCurveBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.horizontal = nn.Conv2d(ch, ch, (1, 7), padding=(0, 3))
        self.vertical = nn.Conv2d(ch, ch, (7, 1), padding=(3, 0))

        self.offset = nn.Conv2d(ch, 18, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.curve = DeformConv2d(ch, ch, 3, padding=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.horizontal(x)
        v = self.vertical(x)
        c = self.curve(x, self.offset(x))
        y = self.fuse(torch.cat([h, v, c], dim=1))
        y = y * self.gate(y)
        return x + y
```

Important details from the code:

- `horizontal` uses a 1x7 convolution to emphasize horizontal strokes and matras.
- `vertical` uses a 7x1 convolution to highlight vertical structures.
- `offset` generates deformation parameters for a deformable convolution, which allows the network to adapt to curved character geometry.
- `curve` is a deformable convolution branch that handles curved edges and irregular letter contours.
- The outputs are concatenated and fused with a 1x1 convolution, then weighted by a learned gate.
- The block adds the input back via a residual connection, preserving the original feature pathway while adding directional refinement.

This design is consistent with the project goal of preserving Bangla characters with long horizontal strokes, vertical pillars, and curved contours.

## Model Architecture and Technical Details

The project is implemented as a generator-discriminator pair. The generator is a multi-scale U-Net-like encoder-decoder with repeated `LineCurveBlock` modules and skip connections.

Key verified architecture details from the notebook:

- Input image shape: `3 x 128 x 256` (RGB, height 128, width 256)
- Batch size: `8`
- Epochs: `20`
- Learning rate: `1e-4`
- Optimizer: Adam with `betas=(0.5, 0.999)`
- Generator channels: `3 -> 64 -> 128 -> 256` with a 4-block bottleneck of `LineCurveBlock(256)`
- Discriminator: a multi-stage convolutional classifier with `64, 128, 256, 512` channels and a final 1-channel output
- Residual skip connections are used between encoder and decoder stages
- Final output is clamped to `[-1, 1]` and added to the input image as a residual restoration term

The generator architecture in simplified form is:

```text
Input (3, 128, 256)
  -> head (7x7 conv)
  -> enc1: LineCurveBlock(64) x2
  -> down1: ConvBlock(64, 128, stride=2)
  -> enc2: LineCurveBlock(128) x2
  -> down2: ConvBlock(128, 256, stride=2)
  -> mid: LineCurveBlock(256) x4
  -> up2 / dec2
  -> up1 / dec1
  -> output conv (3 channels)
  -> residual output + input
```

The discriminator implements a standard adversarial map-based classifier, with each block defined by a `Conv2d` and optional `InstanceNorm2d` + LeakyReLU. The notebook uses a least-squares-style MSE adversarial objective:

```python
d_real = D(clean)
d_fake = D(fake.detach())
d_loss = 0.5 * (
    F.mse_loss(d_real, torch.ones_like(d_real))
    + F.mse_loss(d_fake, torch.zeros_like(d_fake))
)
```

The generator loss is a weighted combination of:

- `L1` reconstruction loss
- edge loss based on Sobel gradients
- VGG-based perceptual loss
- adversarial loss

The exact weights used in the notebook are:

- `W_L1 = 1.0`
- `W_EDGE = 1.0`
- `W_PERC = 0.1`
- `W_ADV = 0.01`

This is a reasonable multi-objective setup for image restoration, where structural fidelity and perceptual realism both matter.

## Dataset and Preprocessing

The notebook uses a paired-image dataset with one clean plate image and two degraded variants: defocus blur and motion blur.

- `GAUSSIAN_DIR = r"D:\_deblur gan\new dataset\light_gaussian_blurred"` (legacy variable/folder name for the defocus-blurred images)
- `MOTION_DIR = r"D:\_deblur gan\new dataset\light_motion_blurred"`
- `CLEAN_DIR = r"D:\_deblur gan\new dataset\fixed cropped"`

The code builds paired entries by matching filenames across folders and resizing each image to `IMG_W = 256` and `IMG_H = 128` using `Image.BICUBIC` interpolation.

The dataset was custom-made for this project, and its labels were manually annotated. The repository does not document the annotation schema or annotation protocol in enough detail to describe those labels more specifically.

The dataset loader does the following:

- loads each image in RGB format
- resizes it to the target dimensions
- converts pixel values from `[0,255]` to `[-1,1]` using `tensor.float() / 127.5 - 1`
- stores the pair in a PyTorch `Dataset`

Train/validation/test splits are created by shuffling all clean filenames and allocating proportions:

- `VAL_SPLIT = 0.1`
- `TEST_SPLIT = 0.1`

The final data arrangement is created with a `DataLoader` for each split. The notebook reports the number of clean images and paired train/validation/test examples at runtime.

Note: the project uses local Windows paths and is not fully portable without updating these paths to match a different machine or environment.

## Training

The training loop is defined directly in the notebook and saves checkpoints for both the latest state and the best validation result. The notebook code sets:

- `CKPT_DIR = "checkpoints_modified_gan"`
- `LAST_PATH = os.path.join(CKPT_DIR, "last.pth")`
- `BEST_PATH = os.path.join(CKPT_DIR, "best.pth")`

The training procedure does the following each epoch:

1. The generator and discriminator are put into training mode.
2. A batch of blurred-clean pairs is passed through the model.
3. The discriminator is trained to classify clean images as real and generated images as fake.
4. The generator is optimized with the combined objective described above.
5. Validation metrics are computed using the model after the epoch.
6. The best checkpoint is selected based on validation PSNR.

The project stores training history in a dictionary with keys:

- `g_loss`
- `d_loss`
- `val_psnr`
- `val_ssim`
- `val_mae`

The notebook plots these values after training and also produces sample visual comparisons of blurry, restored, and clean plates.

Example of the verified training configuration:

```python
CLEAN_DIR = r"D:\_deblur gan\new dataset\fixed cropped"
GAUSSIAN_DIR = r"D:\_deblur gan\new dataset\light_gaussian_blurred"  # legacy name for defocus-blurred images
MOTION_DIR = r"D:\_deblur gan\new dataset\light_motion_blurred"

IMG_H = 128
IMG_W = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
```

The repository does not include a standalone Python training script. Training is executed by opening and running the notebook in a Jupyter environment or VS Code notebook UI.

## Inference and Usage

The repository does not contain a dedicated production inference CLI or script. Inference is handled inside the notebook environment, where the model is loaded from a checkpoint and evaluated on a validation or test loader.

The verified inference pattern in the notebook is:

```python
G.load_state_dict(torch.load(BEST_PATH, map_location=device))
p, s, m = evaluate(G, test_loader)
```

A sample image can also be loaded and passed to the network using the helper function `load_sample_image(...)` and the trained generator. The notebook defines this utility for directly evaluating a single file from the dataset.

For a practical workflow:

1. Open `Line-Curve Block DelburGAN.ipynb` in Jupyter or VS Code.
2. Ensure the dataset path variables match the local machine.
3. Run the training or evaluation cells in order.
4. Use the saved best checkpoint to run restoration on validation or test images.

This repository is not packaged as a CLI tool; the intended workflow is research-oriented notebook execution.

## Experimental Results

The project includes saved evaluation outputs in the `deblurred_outputs` directory.

### Verified summary metrics

The CSV file `deblurred_outputs/per_image_metrics_modified_gan.csv` contains per-image evaluation results for the model. The table below reports the mean values across the saved examples for each blur type.

| Blur type | Blurry PSNR | Deblurred PSNR | Blurry SSIM | Deblurred SSIM | Blurry MAE | Deblurred MAE | PSNR gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Defocus (CSV label: `gaussian`) | 28.9050 | 33.5253 | 0.8899 | 0.9438 | 6.5823 | 4.0031 | +4.6203 |
| Motion | 23.6328 | 29.6682 | 0.7444 | 0.9029 | 11.4394 | 6.0193 | +6.0354 |

These values are based on the project’s saved evaluation CSV and represent the recorded deblurring performance on the local experiment set. They should be interpreted as repository-supported observations, not as global benchmark claims.

The training notebook also reports validation metrics by epoch, including validation PSNR, SSIM, and MAE, and selects the best checkpoint based on the highest validation PSNR. The training log file is available under the local project directories such as `modified_deblur_gan_result/training_log.csv` and `deblur gan v2 new dataset result/training_log.csv`.

Important caveat: because the project uses local folder paths and notebook-defined splits, direct numerical comparisons across experiments should only be made when the same data, preprocessing, and evaluation protocol were used.

### Model Comparison

In addition to the proposed Line-Curve model, baseline models were implemented and trained in separate project notebooks: U-Net, NAFNet, Pix2Pix, DeblurGAN v1, and DeblurGAN v2. U-Net and NAFNet scores are also stored in `deblurred_outputs/per_image_metrics.csv`; scores for the other baselines are taken from the saved per-blur validation summaries in their notebooks. The proposed model's results come from `deblurred_outputs/per_image_metrics_modified_gan.csv` and its notebook.

The tables report per-blur mean PSNR, SSIM, and MAE. PSNR and SSIM are higher-is-better; MAE is lower-is-better.

### Defocus Blur (CSV label: `gaussian`)

| Model | Evaluation split | PSNR (dB) | SSIM | MAE |
| --- | --- | ---: | ---: | ---: |
| U-Net | Validation | 34.40 | 0.9512 | 3.45 |
| NAFNet | Validation | 35.76 | 0.9585 | 3.02 |
| Line-Curve Block model (new) | Test | 33.53 | 0.9438 | 4.00 |
| Pix2Pix | Validation | 29.43 | 0.8922 | 6.08 |
| DeblurGAN v1 | Validation | 29.37 | 0.8788 | 6.13 |
| DeblurGAN v2 | Validation | 31.37 | 0.9246 | 4.78 |

### Motion Blur

| Model | Evaluation split | PSNR (dB) | SSIM | MAE |
| --- | --- | ---: | ---: | ---: |
| U-Net | Validation | 30.24 | 0.9108 | 5.39 |
| NAFNet | Validation | 32.08 | 0.9301 | 4.49 |
| Line-Curve Block model (new) | Test | 29.67 | 0.9029 | 6.02 |
| Pix2Pix | Validation | 25.21 | 0.8031 | 9.52 |
| DeblurGAN v1 | Validation | 25.69 | 0.7844 | 9.23 |
| DeblurGAN v2 | Validation | 27.02 | 0.8510 | 7.59 |

The baseline notebooks evaluate their validation split, while the Line-Curve notebook evaluates its test split. The split labels differ, so treat this as a summary of the saved experiments rather than a strictly controlled, same-split benchmark. The proposed model improves over its blurred input on the saved test examples, but U-Net and NAFNet have higher recorded validation means. These metrics should be considered alongside the qualitative examples below.

## U-Net and NAFNet Error Examples

This comparison shows that higher average PSNR/SSIM does not guarantee every character is restored correctly. These image-wide metrics can overlook localized artifacts or altered strokes that matter for plate readability.

![U-Net and NAFNet restoration errors](figures/unet%20and%20naf%20net%20error.png)


## Installation and Requirements

The repository declares the following dependencies in `requirements.txt`:

```text
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
tqdm>=4.66.0
scikit-image>=0.21.0
scipy>=1.11.0
matplotlib>=3.7.0
ipywidgets>=8.1.0
albumentations>=1.3.1
timm>=0.9.12
tensorboard>=2.15.0
pyyaml>=6.0.1
lpips>=0.1.4
```

The project also references PyTorch and TorchVision, with the requirements file noting that CUDA-enabled installs should be configured separately using a torch wheel index. The notebook imports `torch`, `torch.nn`, `torch.optim`, and `torchvision`-style operations as needed.

Recommended environment:

- Python 3.9+
- CUDA-enabled environment for faster training if available
- Jupyter Notebook or VS Code notebook support
- Windows development environment, based on the hard-coded local dataset paths in the notebooks

---