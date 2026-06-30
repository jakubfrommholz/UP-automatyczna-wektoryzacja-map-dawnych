# Vector from Map — dokumentacja techniczna

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.pl.md">Polski</a> ·
</p>

Wtyczka QGIS dodająca zestaw algorytmów do **automatycznej wektoryzacji map** (Processing Toolbox) oraz szybki dostęp z paska narzędzi.

---

## Spis treści

- [Algorytmy w Processing Toolbox](#algorytmy-w-processing-toolbox) 
- [Biblioteki Python](#biblioteki-python)
- [Instalacja](#instalacja)
- [Integracja](#integracja)
- [Formaty danych](#formaty-danych)
- [Podziękowania](#podziękowania)

---

## Algorytmy w Processing Toolbox

Poniżej lista algorytmów rejestrowanych przez provider (`provider.py`) oraz ich role.

### Diagnostyka

| ID (Processing) | Plik | Opis | Wymagania |
|---|---|---|---|
| `vector_from_map:check_dependencies` | `algorithms/check_dependencies.py` | Raport dostępności bibliotek i wtyczek oraz status kluczowych algorytmów | brak (działa zawsze) |

### Deep learning

| ID (Processing) | Plik | Opis | Wymagania |
|---|---|---|---|
| `vector_from_map:tile_export` | `algorithms/tile_export.py` | Cięcie rastra na kafle + opcjonalna rasteryzacja maski i zapis PNG/GeoTIFF | `cv2` (PNG), GDAL (QGIS), numpy |
| `vector_from_map:pytorch_training` | `algorithms/pytorch_training.py` | Trening segmentacji w QGIS (U-Net lub SMP) | `torch`; dla SMP: `segmentation-models-pytorch` (+ czasem `timm`) |
| `vector_from_map:pth_inference` | `algorithms/pth_inference.py` | Inferencja modelu PyTorch (.pth/.pt) lub ONNX (.onnx) | `torch` lub `onnxruntime` |

### Klasyfikacja

| ID (Processing) | Plik | Opis | Wymagania |
|---|---|---|---|
| `vector_from_map:rgb_classification` | `algorithms/rgb_classification.py` | Klasyfikacja pikseli „najbliższym kolorem” (RGB / CIEDE2000) | numpy |
| `vector_from_map:artificialneuralnetworkclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (ANN) | SAGA Next Gen (`sagang`) |
| `vector_from_map:decisiontreeclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (Decision Tree) | SAGA Next Gen |
| `vector_from_map:logisticregressionclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (Logistic Regression) | SAGA Next Gen |
| `vector_from_map:normalbayesclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (Normal Bayes) | SAGA Next Gen |
| `vector_from_map:randomforestclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (Random Forest) | SAGA Next Gen |
| `vector_from_map:supportvectormachineclassification` | `algorithms/saga_classification.py` | Proxy do SAGA (SVM) | SAGA Next Gen |
| `vector_from_map:kmeansclusteringforrasters` | `algorithms/saga_clustering.py` | Proxy do SAGA (K-Means dla rastrów) | SAGA Next Gen |

### Przetwarzanie obrazu

| ID (Processing) | Plik | Opis | Wymagania |
|---|---|---|---|
| `vector_from_map:edge_detection` | `algorithms/edge_detection.py` | Canny/Sobel (cv2) + opcjonalnie delegacja do algorytmu SAGA „edge/gradient” | `cv2` dla Canny/Sobel; SAGA NG dla metody SAGA |
| `vector_from_map:rasterskeletonization` | `algorithms/saga_image_processing.py` | Proxy do SAGA (skeletonization) | SAGA Next Gen |
| `vector_from_map:region_growing` | `algorithms/region_growing.py` | Seeded Region Growing (BFS od seedów punktowych) | numpy |
| `vector_from_map:split_rgb_bands` | `algorithms/split_rgb_bands.py` | Rozdzielenie rastra RGB (wielopasmowego) na 3 rastry jednopasmowe (R/G/B) | numpy + GDAL (QGIS) |

---

## Biblioteki Python

> QGIS ma własne środowisko Pythona. Instalacja `pip` zależy od dystrybucji QGIS (np. OSGeo4W). Algorytm „Sprawdź zależności” podpowiada brakujące pakiety.

| Pakiet | Wymagany przez | Status w projekcie |
|---|---|---|
| `numpy` | praktycznie wszystkie algorytmy | zakładany jako dostępny w QGIS |
| `opencv-python` (`cv2`) | `tile_export`, `edge_detection` (Canny/Sobel) | wymagany do PNG i filtrów cv2 |
| `torch` | `pytorch_training`, `pth_inference` (PyTorch) | wymagany dla ścieżki .pth/.pt |
| `segmentation-models-pytorch` | `pytorch_training` (modele inne niż U-Net) | wymagany tylko dla SMP |
| `timm` | `pytorch_training` (encodery `timm-*`) | opcjonalny zależnie od encoderów |
| `onnxruntime` | `pth_inference` (ONNX) | wymagany dla ścieżki .onnx |

---

## Instalacja

Skopiuj folder `vector_from_map/` do katalogu pluginów profilu QGIS.

- Windows:
  - `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- Linux/Mac:
  - `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

Następnie uruchom QGIS i włącz plugin:
- `Plugins > Manage and Install Plugins...` → **Vector from Map**

### Instalacja zależności Python (jeżeli potrzebne)

W zależności od dystrybucji QGIS, `pip` trzeba uruchamiać w środowisku Pythona QGIS.
Najprostsza ścieżka weryfikacji:

1. Uruchom algorytm: **Diagnostyka → Sprawdź zależności** (`check_dependencies`).
2. Doinstaluj wskazane pakiety.
---

## Integracje

### Deepness (opcjonalnie)

W `tile_export.py` wtyczka korzysta z Deepness, jeśli jest zainstalowany:
- zaokrągla extent wejściowego rastra do siatki pikseli przez
  `deepness.processing.extent_utils.round_extent_to_rlayer_grid(extent, rlayer)`.

Fallback:
- gdy Deepness nie jest dostępny, algorytm działa w pełni samodzielnie.

#### Krytyczne (stabilność QGIS)

W `SagaProxyAlgorithm.initAlgorithm()` **nie wolno** iterować po `algo.outputDefinitions()` i robić `self.addOutput(o)`. Outputy typu *destination* są rejestrowane automatycznie przez `addParameter(p.clone())`.

Dodanie obcych obiektów C++ jako outputów może prowadzić do **use-after-free** i crash QGIS.

---

## Formaty danych

### Tile Export (`tile_export`)

- kafle PNG: `tile_img_{col}_{row}.png`
- kafle GeoTIFF: `tile_img_{col}_{row}.tif`
- maski PNG (jeżeli podano warstwę maski): `tile_mask_{col}_{row}.png`

Opcjonalnie dane trafiają do podfolderu `DDMMYYYY_HHMMSS`.

Maska jest rasteryzowana do wartości 0/1 (burn=1) na geometrii wejściowej warstwy wektorowej.

### Trening PyTorch (`pytorch_training`)

- wejście: raster obrazowy + raster maski o tym samym extent i rozdzielczości
- maska: wartości klas w paśmie 1 w zakresie `0..NUM_CLASSES-1`
- `nodata` oraz wartości spoza zakresu → `ignore_index=-1`
- patching: patch size musi być podzielny przez 16; stride dowolny ≥ 1

Wyjście: plik `.pth` jako checkpoint:

```python
{
  'format': 'vector_from_map_torch_checkpoint_v1',
  'model_type': 'unet' | 'deeplabv3plus' | 'linknet' | 'fpn' | 'pspnet' | 'manet',
  'encoder_name': 'resnet34' | ... (dla SMP),
  'in_channels': int,
  'num_classes': int,
  'base_channels': int,
  'state_dict': ...,
}
```

### Inferencja (`pth_inference`)

- format modelu: auto po rozszerzeniu lub wybór ręczny
- kafelkowanie: `TILE_SIZE` i `OVERLAP`
  - gdy `OVERLAP > 0` logity są **uśredniane** (akumulator + licznik), a dopiero potem wykonywany jest argmax / threshold / softmax

Typy wyjścia:
- **Indeksy klas (Int32)**: `argmax` po wymiarze klas
- **Sigmoid binary (UInt8)**: 1 kanał + próg 0.5 (sigmoid stosowany automatycznie, jeśli wartości wyglądają na logity)
- **Softmax probability (Float32)**: max prawdopodobieństwo (dla 1 kanału sigmoid)

### Klasyfikacja RGB (`rgb_classification`)

- wejście: raster ≥ 3 pasma
- definicja klas: tabela `[Nazwa, R, G, B]`
- metryki:
  - euklidesowa w RGB
  - CIEDE2000 (sRGB→Lab D65 + ΔE2000; implementacja w `algorithms/_cie.py`)

Wyjście: GeoTIFF z indeksami klas (Byte gdy ≤255 klas, inaczej Int16).

### Split RGB Bands (`split_rgb_bands`)

- wejście: raster min. 3 pasma
- wyjście: 3 GeoTIFF (LZW) — pasma 1,2,3 jako R,G,B
- typ danych i nodata są zachowane bez konwersji

### Seeded Region Growing (`region_growing`)

- wejście: raster (dla RGB używana średnia z 1..3 pasma), warstwa punktowa seedów
- opcjonalnie `CLASS_FIELD`: seedy z tym samym class_id tworzą jeden region
- wynik: GeoTIFF Int32, `nodata=-1`, wartości = `class_id` lub `-1`

---

Przykład:

```python
class MyNewSagaAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:somealgo'
    DISPLAY_NAME = 'Moje (SAGA)'
    GROUP_NAME = 'SAGA — ...'
    GROUP_ID = 'saga_...'
```

---

## Podziękowania
GeoSegStudio (https://github.com/dronnix-io/GeoSegStudio)
GeoAI: Artificial Intelligence for Geospatial Data (https://github.com/opengeos/geoai)

Projekt wykonany w ramach projektu ["Uczelnie Przyszłości"](https://www.gov.pl/web/ncbr/projekt-uczelnie-przyszlosci).
