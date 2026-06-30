# Vector from Map — Technical Documentation

<p align="center">
  <a href="README.pl.md">Polski</a>
</p>

A QGIS plugin that adds a set of algorithms for **automatic map vectorization** (Processing Toolbox) and quick access from the toolbar.

---

## Table of Contents

- [Algorithms in Processing Toolbox](#algorithms-in-processing-toolbox)
- [Python Libraries](#python-libraries)
- [Installation](#installation)
- [Integrations](#integrations)
- [Data Formats](#data-formats)
- [Acknowledgments](#acknowledgments)

---

## Algorithms in Processing Toolbox

Below is a list of algorithms registered by the provider (provider.py) and their roles.

### Diagnostics

| ID (Processing) | File | Description | Requirements |
|---|---|---|---|
| vector_from_map:check_dependencies | algorithms/check_dependencies.py | Report of library and plugin availability and status of key algorithms | none (always works) |

### Deep Learning

| ID (Processing) | File | Description | Requirements |
|---|---|---|---|
| vector_from_map:tile_export | algorithms/tile_export.py | Raster tiling into tiles + optional mask rasterization and PNG/GeoTIFF export | cv2 (PNG), GDAL (QGIS), numpy |
| vector_from_map:pytorch_training | algorithms/pytorch_training.py | Segmentation training in QGIS (U-Net or SMP) | torch; for SMP: segmentation-models-pytorch (+ sometimes timm) |
| vector_from_map:pth_inference | algorithms/pth_inference.py | Inference of PyTorch model (.pth/.pt) or ONNX (.onnx) | torch or onnxruntime |

### Classification

| ID (Processing) | File | Description | Requirements |
|---|---|---|---|
| vector_from_map:rgb_classification | algorithms/rgb_classification.py | Pixel classification by "nearest color" (RGB / CIEDE2000) | numpy |
| vector_from_map:artificialneuralnetworkclassification | algorithms/saga_classification.py | SAGA proxy (ANN) | SAGA Next Gen (sagang) |
| vector_from_map:decisiontreeclassification | algorithms/saga_classification.py | SAGA proxy (Decision Tree) | SAGA Next Gen |
| vector_from_map:logisticregressionclassification | algorithms/saga_classification.py | SAGA proxy (Logistic Regression) | SAGA Next Gen |
| vector_from_map:normalbayesclassification | algorithms/saga_classification.py | SAGA proxy (Normal Bayes) | SAGA Next Gen |
| vector_from_map:randomforestclassification | algorithms/saga_classification.py | SAGA proxy (Random Forest) | SAGA Next Gen |
| vector_from_map:supportvectormachineclassification | algorithms/saga_classification.py | SAGA proxy (SVM) | SAGA Next Gen |
| vector_from_map:kmeansclusteringforrasters | algorithms/saga_clustering.py | SAGA proxy (K-Means for rasters) | SAGA Next Gen |

### Image Processing

| ID (Processing) | File | Description | Requirements |
|---|---|---|---|
| vector_from_map:edge_detection | algorithms/edge_detection.py | Canny/Sobel (cv2) + optional delegation to SAGA "edge/gradient" algorithm | cv2 for Canny/Sobel; SAGA NG for SAGA method |
| vector_from_map:rasterskeletonization | algorithms/saga_image_processing.py | SAGA proxy (skeletonization) | SAGA Next Gen |
| vector_from_map:region_growing | algorithms/region_growing.py | Seeded Region Growing (BFS from point seeds) | numpy |
| vector_from_map:split_rgb_bands | algorithms/split_rgb_bands.py | Split RGB raster (multiband) into 3 single-band rasters (R/G/B) | numpy + GDAL (QGIS) |

---

## Python Libraries

> QGIS has its own Python environment. pip installation depends on the QGIS distribution (e.g., OSGeo4W). The "Check Dependencies" algorithm suggests missing packages.

| Package | Required by | Project Status |
|---|---|---|
| numpy | virtually all algorithms | assumed available in QGIS |
| opencv-python (cv2) | tile_export, edge_detection (Canny/Sobel) | required for PNG and cv2 filters |
| torch | pytorch_training, pth_inference (PyTorch) | required for .pth/.pt path |
| segmentation-models-pytorch | pytorch_training (models other than U-Net) | required only for SMP |
| timm | pytorch_training (timm-* encoders) | optional depending on encoders |
| onnxruntime | pth_inference (ONNX) | required for .onnx path |

---

## Installation

Copy the `vector_from_map/` folder to the QGIS profile plugins directory.

- Windows:
  - `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- Linux/Mac:
  - `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

Then launch QGIS and enable the plugin:
- Plugins > Manage and Install Plugins... → **Vector from Map**

### Installing Python Dependencies (if needed)

Depending on the QGIS distribution, pip must be run in the QGIS Python environment.
The simplest verification path:

1. Run the algorithm: **Diagnostics → Check Dependencies** (check_dependencies).
2. Install the indicated packages.

---

## Integrations

### Deepness (optional)

In `tile_export.py` the plugin uses Deepness if installed:
- rounds the extent of the input raster to the pixel grid via
  `deepness.processing.extent_utils.round_extent_to_rlayer_grid(extent, rlayer)`.

Fallback:
- when Deepness is not available, the algorithm works fully independently.

#### Critical (QGIS Stability)

In `SagaProxyAlgorithm.initAlgorithm()` **do not** iterate over `algo.outputDefinitions()` and call `self.addOutput(o)`. Outputs of type *destination* are registered automatically via `addParameter(p.clone())`.

Adding foreign C++ objects as outputs can lead to **use-after-free** and QGIS crashes.

---

## Data Formats

### Tile Export (tile_export)

- PNG tiles: `tile_img_{col}_{row}.png`
- GeoTIFF tiles: `tile_img_{col}_{row}.tif`
- PNG masks (if mask layer provided): `tile_mask_{col}_{row}.png`

Optionally data goes into a subfolder `DDMMYYYY_HHMMSS`.

The mask is rasterized to values 0/1 (burn=1) on the input geometry of the vector layer.

### PyTorch Training (pytorch_training)

- input: image raster + mask raster with same extent and resolution
- mask: class values in band 1 in range `0..NUM_CLASSES-1`
- nodata and values outside range → `ignore_index=-1`
- patching: patch size must be divisible by 16; stride any ≥ 1

Output: .pth file as checkpoint:

```python
{
  'format': 'vector_from_map_torch_checkpoint_v1',
  'model_type': 'unet' | 'deeplabv3plus' | 'linknet' | 'fpn' | 'pspnet' | 'manet',
  'encoder_name': 'resnet34' | ... (for SMP),
  'in_channels': int,
  'num_classes': int,
  'base_channels': int,
  'state_dict': ...,
}
```
### Inference (pth_inference)

- model format: auto by extension or manual selection
- tiling: TILE_SIZE and OVERLAP
  - when OVERLAP > 0 logits are **averaged** (accumulator + counter), then argmax / threshold / softmax is performed

Output types:
- **Class Indices (Int32)**:  rgmax over class dimension
- **Sigmoid Binary (UInt8)**: 1 channel + threshold 0.5 (sigmoid applied automatically if values look like logits)
- **Softmax Probability (Float32)**: max probability (for 1 channel sigmoid)

### RGB Classification (
gb_classification)

- input: raster ≥ 3 bands
- class definition: table [Name, R, G, B]
- metrics:
  - Euclidean in RGB
  - CIEDE2000 (sRGB→Lab D65 + ΔE2000; implementation in  lgorithms/_cie.py)

Output: GeoTIFF with class indices (Byte if ≤255 classes, otherwise Int16).

### Split RGB Bands (split_rgb_bands)

- input: raster min. 3 bands
- output: 3 GeoTIFF (LZW) — bands 1,2,3 as R,G,B
- data type and nodata are preserved without conversion

### Seeded Region Growing (
egion_growing)

- input: raster (for RGB the average of bands 1..3 is used), point layer of seeds
- optionally CLASS_FIELD: seeds with the same class_id form one region
- result: GeoTIFF Int32, 
odata=-1, values = class_id or -1

---

## Example

\\\python
class MyNewSagaAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:somealgo'
    DISPLAY_NAME = 'My Algorithm (SAGA)'
    GROUP_NAME = 'SAGA — ...'
    GROUP_ID = 'saga_...'
\\\

---

## Acknowledgments

GeoSegStudio (https://github.com/dronnix-io/GeoSegStudio)
GeoAI: Artificial Intelligence for Geospatial Data (https://github.com/opengeos/geoai)

Project implemented as part of the ["Uczelnie Przyszłości"](https://www.gov.pl/web/ncbr/projekt-uczelnie-przyszlosci) project.
