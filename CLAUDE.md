# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# vectorization_bridge — QGIS Plugin

Wtyczka QGIS z integracją innych wtyczek (Deepness, GeoAI, SAGA NG). Rejestruje **16 algorytmów** w Processing Toolbox (8 własnych + 8 cienkich proxy do SAGA). Wszystkie integracje są opcjonalne — wtyczka działa w pełni bez nich (proxy SAGA są widoczne, ale zwracają błąd przy uruchomieniu, gdy SAGA NG nie jest zainstalowane).

**Grupy w Processing Toolbox** (po reorganizacji):
- `Diagnostyka` — check_dependencies
- `Deep learning` — tile_export, pytorch_training (U-Net + SMP), pth_inference (PyTorch + ONNX)
- `Klasyfikacja` — rgb_classification, 6× SAGA classification, SAGA k-means
- `Przetwarzanie obrazu` — edge_detection, SAGA skeletonization, Watershed, Seeded Region Growing

**KRYTYCZNE w `SagaProxyAlgorithm.initAlgorithm()`**: NIE wolno iterować `algo.outputDefinitions()` i robić `self.addOutput(o)` — destination params (sklonowane przez `addParameter(p.clone())`) automatycznie rejestrują outputy. Jednoczesne dodanie cudzych obiektów C++ powoduje use-after-free → crash QGIS.

W toolbarze QGIS (`iface.pluginToolBar()`) pojawia się ikona z rozwijanym menu (`QToolButton.InstantPopup`) grupującym algorytmy tematycznie. Każda akcja menu otwiera natywny `processing.execAlgorithmDialog(...)`.

## Struktura

```
vectorization_bridge/
├── __init__.py              # classFactory → VectorizationBridgePlugin
├── metadata.txt             # hasProcessingProvider=yes, qgisMinimumVersion=3.16, icon=icon.svg
├── icon.svg                 # ikona wtyczki (toolbar + menu Wtyczki + provider.icon())
├── plugin.py                # provider + QToolButton z QMenu w toolbarze
├── provider.py              # VectorizationBridgeProvider — loadAlgorithms() rejestruje 16 algo + icon()
├── ciede-2000.py            # public-domain skalarna referencja CIEDE2000 (NIE importowane — myślnik w nazwie)
├── qgis-plugin-deepness-devel/  # źródła Deepness — TYLKO podgląd, w .gitignore, nie wgrywane do paczki
├── .gitignore               # wyklucza qgis-plugin-deepness-devel/ i __pycache__
└── algorithms/
    ├── __init__.py
    ├── _compat.py           # CENTRALNY moduł detekcji zależności — importuj stąd
    ├── _cie.py              # rgb_to_lab (sRGB→Lab D65) + zwektoryzowany numpy CIEDE2000
    ├── _torch_models.py     # build_model(...) — wspólna definicja modeli dla treningu i inferencji
    ├── tile_export.py       # TileExportAlgorithm — integracja z Deepness
    ├── pytorch_training.py  # PytorchTrainingAlgorithm — trening segmentacji PyTorch (U-Net + SMP)
    ├── pth_inference.py     # PthInferenceAlgorithm — informuje o Deepness (ONNX) i GeoAI
    ├── edge_detection.py    # EdgeDetectionAlgorithm — integracja z SAGA NG
    ├── rgb_classification.py # RgbClassificationAlgorithm — Euklidesowa + CIEDE2000
    ├── check_dependencies.py # CheckDependenciesAlgorithm — diagnostyka zależności
    ├── saga_proxy.py        # SagaProxyAlgorithm — klasa bazowa proxy do sagang:*
    ├── saga_classification.py # 6 podklas (ANN, DT, LR, NB, RF, SVM) — grupa "Klasyfikacja"
    ├── saga_clustering.py   # SagaKMeansClusteringAlgorithm — grupa "Klasyfikacja"
    ├── saga_image_processing.py # SagaRasterSkeletonizationAlgorithm — grupa "Przetwarzanie obrazu"
    ├── watershed.py         # WatershedAlgorithm — cv2.watershed + auto markery (dist transform)
    └── region_growing.py    # SeededRegionGrowingAlgorithm — BFS od seedów wektorowych
```

## _compat.py — API

Wszystkie sprawdzenia dostępności wtyczek/bibliotek muszą korzystać z tego modułu.

```python
from ._compat import (
    deepness_available, deepness_version,
    deepness_round_extent,          # (extent, rlayer) → QgsRectangle
    geoai_available, geoai_version,
    saga_available,
    saga_find_edge_algorithm,       # () → (algo_id, display_name) lub (None, None)
    saga_find_rf_algorithm,         # () → (algo_id, display_name) lub (None, None)
    cv2_available, cv2_version,
    torch_available, torch_version,
    smp_available, smp_version,
    timm_available, timm_version,
    onnxruntime_available, onnxruntime_version,
)
```

## Integracje z wtyczkami

### Deepness (opcjonalny)

- **Gdzie:** `tile_export.py`
- **Co:** zaokrąglenie extent do siatki pikseli rastra przed cięciem na kafle
- **Funkcja:** `deepness.processing.extent_utils.round_extent_to_rlayer_grid(extent, rlayer)`
- **Fallback:** brak zaokrąglenia (extent używany wprost)
- **Uwaga:** Deepness obsługuje modele ONNX; ten plugin obsługuje .pth — są komplementarne

### SAGA NG (opcjonalny)

- **Dwa wzorce delegacji** współistnieją:
  1. **`edge_detection.py`** — dynamiczne wykrywanie SAGA edge-algorytmu i dodanie go jako opcji enum metody. Metody 0,1 = cv2 (Canny/Sobel), metody ≥ 2 = SAGA przez `processing.run(saga_id, params, ...)` z heurystycznym mapowaniem kluczy parametrów.
  2. **`saga_proxy.py` + 9 podklas** — `SagaProxyAlgorithm` klonuje `parameterDefinitions()` i `outputDefinitions()` algorytmu SAGA w `initAlgorithm()`, a w `processAlgorithm()` forwarduje parametry 1:1 przez `processing.run(SAGA_ALGO_ID, parameters)`. ID algorytmu = `vectorization_bridge:<saga_short_name>` (np. `randomforestclassification`).
- **Gdy SAGA niedostępne:** proxy są widoczne, ale `checkParameterValues` zwraca komunikat o konieczności instalacji.

### GeoAI (opcjonalny, informacyjnie)

- **Gdzie:** `pth_inference.py`, `check_dependencies.py`
- **Co:** jedynie informacja w logach — brak delegacji do GeoAI

## Zależności

| Biblioteka | Wymagana przez | Uwagi |
|---|---|---|
| numpy | wszystkie | wbudowane w QGIS, nie sprawdzać |
| opencv-python (cv2) | tile_export, edge_detection | soft-fail w checkParameterValues |
| torch (PyTorch) | pytorch_training, pth_inference (gałąź .pth/.pt) | soft-fail w checkParameterValues |
| segmentation-models-pytorch | pytorch_training (DeepLabV3Plus/LinkNet/FPN/PSPNet/MAnet) | wymagane dla modeli SMP |
| timm | pytorch_training (encodery `timm-*`) | wymagane dla części backbone'ów |
| onnxruntime | pth_inference (gałąź .onnx) | soft-fail w checkParameterValues |
| osgeo (gdal, ogr, osr) | wszystkie | wbudowane w QGIS |

### Wtyczki QGIS — etykiety statusu

- **SAGA Next Gen** — *WYMAGANA* dla pełnej funkcjonalności (9 algo: edge_detection + 8 proxy).
- **GeoAI** — *ZALECANA* (opcjonalna), do trenowania modeli.
- **Deepness** — *opcjonalna*, integracja: zaokrąglanie extent w tile_export, komplementarne UI dla ONNX.

## Wzorce

### Zapis wyjściowego rastra (GDAL)

```python
ds = gdal.GetDriverByName('GTiff').Create(path, w, h, 1, gdal_dtype, options=['COMPRESS=LZW'])
ds.SetGeoTransform((extent.xMinimum(), px, 0, extent.yMaximum(), 0, -py))
srs = osr.SpatialReference(); srs.ImportFromWkt(layer.crs().toWkt())
ds.SetProjection(srs.ExportToWkt())
```

### QgsRasterBlock → numpy

```python
dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
arr = np.frombuffer(bytes(blk.data()), dtype=dt_map.get(blk.dataType(), np.float32)).reshape(ah, aw)
```

### Siatka kafli

```python
stride = max(tile_size - overlap, 1)
x_bins = 1 if raster_w <= tile_size else math.ceil((raster_w - tile_size) / stride) + 1
```

### SagaProxyAlgorithm — dodanie nowego proxy

```python
class MyNewSagaAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:somealgo'
    DISPLAY_NAME = 'Moje (SAGA)'
    GROUP_NAME = 'SAGA — ...'
    GROUP_ID = 'saga_...'
```

Następnie zaimportować i `addAlgorithm` w [provider.py](provider.py) **oraz** dodać wpis w `_MENU_GROUPS` w [plugin.py](plugin.py) (id = `vectorization_bridge:<saga_short_name>`).

## Algorytmy — szczegóły

### tile_export.py — TileExportAlgorithm

- **Wyjście (Deepness-style naming):** `tile_img_{col}_{row}.png` (gdy `EXPORT_IMAGES`) + `tile_img_{col}_{row}.tif` (gdy `EXPORT_GEOTIFF`) + `tile_mask_{col}_{row}.png` (gdy podana `MASK_LAYER`)
- **Podkatalog timestamp:** gdy `USE_TIMESTAMP_SUBDIR=True` — dane trafiają do `OUTPUT_FOLDER/DDMMYYYY_HHMMSS/`
- **Deepness:** jeśli dostępny — wywołuje `deepness_round_extent(raw_extent, layer)` przed iteracją kafli

### pth_inference.py — PthInferenceAlgorithm (PyTorch + ONNX)

- **Format modelu:** auto-detekcja po rozszerzeniu (`.pth/.pt` → torch; `.onnx` → onnxruntime); ręczny override przez `MODEL_FORMAT`.
- **PyTorch:** `torch.load(weights_only=False)` → fallback `torch.jit.load()`.
- **Checkpointy treningowe:** obsługa checkpointu `state_dict` dla `model_type` (`unet`, `deeplabv3plus`, `linknet`, `fpn`, `pspnet`, `manet`) z odtworzeniem architektury przez `_torch_models.build_model`.
- **ONNX:** `onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])`; wzorzec z [Deepness model_base.py:37](qgis-plugin-deepness-devel/qgis-plugin-deepness-devel/src/deepness/processing/models/model_base.py#L37) (do podglądu w katalogu wtyczki, NIE wgrywany — patrz `.gitignore`).
- **OVERLAP > 0:** akumulator logitów `(C, H, W)` + counter `(H, W)`; postprocessing po stitching.
- **OUTPUT_TYPE:** Indeksy klas (Int32, nodata=-1) | Sigmoid binary (UInt8 0/1) | Softmax probability (Float32).
- **Runner:** lokalna funkcja `_load_runner(fmt, path, feedback)` zwraca callable `runner(tile_chw_float32) -> ndarray`; ujednolica obie gałęzie. Wyjście modelu znormalizowane do `(C, H, W)` przez `_normalize_output()`.

### pytorch_training.py — PytorchTrainingAlgorithm (PyTorch, U-Net + SMP)

- **Model (enum):** parametr `MODEL_ARCH` (dropdown): `U-Net`, `DeepLabV3Plus`, `LinkNet`, `FPN`, `PSPNet`, `MAnet`.
- **Encoder (enum):** parametr `ENCODER` (backbone) dla modeli SMP: `resnet18`, `resnet34`, `resnet50`, `vgg16`, `vgg19`, `densenet121`, `densenet169`, `mobilenet_v3_large`, `mobilenet_v3_small`, `timm-gandalf`, `timm-efficientnet-b0`.
- **Wejścia:** raster obrazowy (`INPUT`) + raster masek (`MASK`, klasy w paśmie 1), z kontrolą zgodności rozmiaru i extent.
- **Patching:** trening na patchach (`PATCH_SIZE`, `STRIDE`), odrzucanie patchy bez poprawnych etykiet.
- **Etykiety:** klasy `0..NUM_CLASSES-1`; nodata i wartości spoza zakresu są mapowane na `ignore_index=-1`.
- **Trening:** `CrossEntropyLoss(ignore_index=-1)` + `Adam`; opcjonalne CUDA (`USE_CUDA`).
- **Wyjście:** checkpoint `.pth` (`state_dict` + metadane: `model_type`, `encoder_name`, `in_channels`, `num_classes`, `base_channels`) zgodny z `pth_inference.py`.

### edge_detection.py — EdgeDetectionAlgorithm

- **Metody:** budowane dynamicznie przez `_build_method_list()` przy każdym wywołaniu
- **SAGA:** delegacja przez `processing.run()` gdy method index ≥ 2
- **cv2:** Canny (uint8) / Sobel magnitude (float32) dla metod 0 i 1

### rgb_classification.py — RgbClassificationAlgorithm

- **Parametr COLORS:** flat list `[Nazwa, R, G, B, ...]`; parsowanie `int(float(raw[i*4+k]))`
- **TILE_SIZE wewnętrzny:** 2048 px
- **Metryki:** `METRIC_EUCLIDEAN=0` (na RGB) lub `METRIC_CIEDE2000=1` (sRGB→Lab D65, wzór CIE ΔE2000)
- **CIEDE2000:** korzysta z `_cie.rgb_to_lab` + `_cie.ciede2000`; klasyfikacja iteruje po N klasach (bez alokacji `H×W×N×3`)

### saga_proxy.py — SagaProxyAlgorithm + 9 podklas

- **Klonowanie parametrów:** `for p in algo.parameterDefinitions(): self.addParameter(p.clone())` (parametry nie mogą być współdzielone między algorytmami).
- **Forward 1:1:** `processing.run(self.SAGA_ALGO_ID, parameters, context=context, feedback=feedback)` — klucze parametrów = klucze SAGA, bez mapowania.
- **Brak SAGA:** `initAlgorithm()` cicho nie dodaje parametrów; `checkParameterValues()` zwraca jasny komunikat błędu.

### watershed.py — WatershedAlgorithm

- **Pipeline (cv2):** grayscale → GaussianBlur → Otsu/adapt thresh → distanceTransform → próg `DIST_THRESHOLD × max` → connectedComponents → `cv2.watershed(rgb, markers)`.
- **Markery automatyczne** (bez interakcji); użytkownik kontroluje binaryzację i próg distance.
- **Wyjście:** Int32, nodata=−1; granice = −1, tło = 1, regiony = 2..N.
- **Limit pamięci:** wymaga (H, W, 3) uint8 w pamięci — dla rastrów >5000×5000 px ostrzec.

### region_growing.py — SeededRegionGrowingAlgorithm

- **BFS w 4-sąsiedztwie**, klasyczny SRG (Adams & Bischof 1994).
- **Seedy:** warstwa wektorowa punktowa; `CLASS_FIELD` opcjonalne (puste = każdy punkt = osobny region; podane = łączenie po `class_id`).
- **Kryterium dołączenia:** `|piksel − średnia regionu| ≤ TOLERANCE`. Średnia jest aktualizowana po każdym dołączeniu.
- **Konflikt:** wygrywa pierwszy region (kolejność BFS).
- **CRS:** `QgsCoordinateTransform(src_crs, raster_crs, QgsProject.instance())` — seedy transformowane do CRS rastra przed mapowaniem na piksele.
- **Wyjście:** Int32, nodata=−1.

### check_dependencies.py — CheckDependenciesAlgorithm

- **Parametr VERBOSE:** gdy True — listuje algorytmy SAGA z 'edge'/'gradient'/'filter'/'deriv' w nazwie
- **Status algorytmów:** zawiera także wpis `Trenowanie modelu — PyTorch (U-Net + SMP)`.
- **Wyjście REPORT:** string z pełnym raportem (też w panelu Processing)

### _cie.py — narzędzia barw

- `rgb_to_lab(rgb)` — sRGB (0–255) → CIE Lab przy iluminancie D65, macierz Lindblooma; zwektoryzowana po dowolnym shape `(..., 3)`.
- `ciede2000(lab1, lab2)` — różnica barw, zwektoryzowane przeniesienie skalarnej referencji z `ciede-2000.py`. Public-domain.

## Plugin GUI — toolbar + menu

W [plugin.py](plugin.py):
- `QToolButton` z `setPopupMode(QToolButton.InstantPopup)` i `setMenu(menu)` jest dodawany do `iface.pluginToolBar().addWidget(...)`. `iface.addToolBarIcon(QAction)` **nie** działa dla rozwijanych menu (QAction.setMenu jest ignorowane w toolbarze).
- `_MENU_GROUPS` to lista `(etykieta_grupy, [(algo_id, label), ...])`. Każdy `triggered` woła `processing.execAlgorithmDialog(algo_id, {})`.
- `unload()` musi usunąć akcję z `pluginToolBar()` przez `toolbar.removeAction(self.toolbar_action)`, gdzie `toolbar_action` to wartość zwrócona przez `addWidget(...)`.

## Instalacja

Skopiuj folder `vectorization_bridge/` do:
- Linux/Mac: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
- Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`

Następnie włącz w QGIS: `Plugins > Manage and Install Plugins`.

Aby sprawdzić status integracji, uruchom algorytm `Diagnostyka > Sprawdź zależności` (lub kliknij ikonę w toolbarze → `Diagnostyka > Sprawdź zależności`).
