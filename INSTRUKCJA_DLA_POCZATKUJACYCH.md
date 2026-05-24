# Vectorization Bridge — instrukcja dla początkującego

Ta instrukcja opisuje **co robi każda opcja** dostępna w menu/toolbarze wtyczki Vectorization Bridge oraz **co oznaczają parametry** w oknach algorytmów Processing.

## Gdzie znaleźć opcje wtyczki

Po instalacji i włączeniu wtyczki zobaczysz:

1. **Ikonę na pasku narzędzi** (toolbar) — kliknięcie od razu rozwija menu (bez dodatkowego okna).
2. **Menu `Wtyczki → Vectorization Bridge`** — to samo menu.
3. **Processing Toolbox** — dostawca (provider) `Vectorization Bridge` z pogrupowanymi algorytmami.

Każda pozycja z menu otwiera standardowe okno **Processing → parametry algorytmu**.

## Szybki start (polecana kolejność)

1. **Diagnostyka → Sprawdź zależności**: najpierw sprawdź, czy masz zainstalowane wymagane dodatki (SAGA/cv2/torch/onnxruntime).
2. Jeśli robisz ML:
   - **Cięcie rastra na kafle (ML)** → przygotowanie danych.
   - **Trenowanie modelu (PyTorch)** → trening na rastrze i masce.
   - **Inferencja modelu (PyTorch / ONNX)** → uruchomienie modelu na całym rastrze.
3. Jeśli robisz klasyfikację „po kolorach”: **Klasyfikacja pikseli RGB**.
4. Jeśli robisz segmentację od punktów startowych: **Seeded Region Growing**.
5. Jeśli potrzebujesz narzędzi SAGA: upewnij się, że wtyczka **SAGA Next Gen** jest zainstalowana.

---

# DIAGNOSTYKA

## Sprawdź zależności

**Do czego służy:** generuje raport (w oknie wyników Processing), czy środowisko QGIS ma biblioteki i wtyczki wymagane/zalecane przez Vectorization Bridge.

**Parametry (opcje):**
- **Szczegółowy raport (lista algorytmów SAGA)** (`VERBOSE`)
  - `NIE` (domyślnie): krótki raport.
  - `TAK`: dodatkowo wypisuje przykładowe algorytmy SAGA związane z krawędziami/gradientem (pomocne w diagnozie integracji).

**Co warto wiedzieć:**
- **SAGA Next Gen** jest praktycznie **wymagana do pełnej funkcjonalności** (proxy SAGA + metoda SAGA w wykrywaniu krawędzi).
- `opencv-python (cv2)` jest wymagane m.in. do eksportu PNG w kafelkach oraz do Canny/Sobel.
- `torch` jest wymagany do modeli `.pth/.pt`, a `onnxruntime` do `.onnx`.

---

# DEEP LEARNING

## Cięcie rastra na kafle (ML)

**Do czego służy:** dzieli raster na małe obrazki (kafle) do uczenia maszynowego. Może też zapisać kafle GeoTIFF z georeferencją i maski (z warstwy wektorowej).

**Parametry (opcje):**
- **Wejściowy raster** (`INPUT`)
  - Raster, który chcesz pociąć na kafle.
- **Rozmiar kafla (piksele)** (`TILE_SIZE`)
  - Typowo 256 lub 512.
  - Im większy kafel, tym więcej RAM i wolniej, ale więcej kontekstu dla modelu.
- **Zakładka (piksele)** (`OVERLAP`)
  - Ile pikseli ma nachodzić jeden kafel na drugi.
  - Używaj, gdy potem chcesz minimalizować artefakty na granicach kafli.
  - Musi być **mniejsze niż rozmiar kafla**.
- **Warstwa maski wektorowej (opcjonalna)** (`MASK_LAYER`)
  - Warstwa wektorowa (poligony/linie/punkty) rasteryzowana do maski 0/1.
  - Gdy nie podasz maski — nie powstaną pliki `tile_mask_...png`.
- **Zapisz kafle obrazów (PNG)** (`EXPORT_IMAGES`)
  - `TAK`: zapisuje `tile_img_{col}_{row}.png`.
- **Zapisz kafle GeoTIFF (z georeferencją)** (`EXPORT_GEOTIFF`)
  - `TAK`: zapisuje `tile_img_{col}_{row}.tif` (przydatne do kontroli w QGIS).
- **Utwórz podkatalog z timestampem (DDMMYYYY_HHMMSS)** (`USE_TIMESTAMP_SUBDIR`)
  - `TAK` (domyślnie): każde uruchomienie trafia do osobnego folderu.
  - `NIE`: zapis do wskazanego folderu bez podkatalogu.
- **Folder wyjściowy** (`OUTPUT_FOLDER`)
  - Katalog, w którym pojawią się kafle.

**Wyjście (pliki):**
- `tile_img_{col}_{row}.png` (jeśli PNG włączone)
- `tile_img_{col}_{row}.tif` (jeśli GeoTIFF włączony)
- `tile_mask_{col}_{row}.png` (jeśli podano `MASK_LAYER`)

**Najczęstsze problemy:**
- Błąd o braku `cv2` → doinstaluj `opencv-python`.
- `OVERLAP >= TILE_SIZE` → zmniejsz zakładkę lub zwiększ kafel.

---

## Trenowanie modelu (PyTorch, U-Net)

**Do czego służy:** trenuje model segmentacji/klasyfikacji pikseli na rastrze wejściowym i rastrze masek.

**Ważne pojęcia (prosto):**
- **Patch**: mały wycinek rastra, na którym model się uczy.
- **Epoch (epoka)**: jedno przejście po zbiorze treningowym.
- **Batch**: ile patchy naraz jest liczone na GPU/CPU.

**Parametry (opcje):**
- **Raster wejściowy (obrazy treningowe)** (`INPUT`)
  - Najczęściej RGB, ale może być wielokanałowy.
- **Raster masek (klasy w paśmie 1)** (`MASK`)
  - W paśmie 1 powinny być liczby: `0..(N-1)`.
  - Nodata i wartości spoza zakresu są ignorowane podczas treningu.
- **Model** (`MODEL_ARCH`)
  - `U-Net` (wbudowany) lub modele z biblioteki `segmentation-models-pytorch`.
- **Encoder (backbone, dla modeli SMP)** (`ENCODER`)
  - Wybór „kręgosłupa” sieci dla modeli SMP (np. `resnet34`).
  - Dla U-Net ten parametr nie ma praktycznego znaczenia (ale zostaje w UI).
- **Liczba kanałów wejściowych** (`INPUT_CHANNELS`)
  - Ile pasm model ma czytać z rastra (np. 3 dla RGB).
- **Liczba klas** (`NUM_CLASSES`)
  - Ile klas przewiduje model (minimum 2).
- **Rozmiar patcha (piksele)** (`PATCH_SIZE`)
  - Musi być **podzielny przez 16** (np. 256, 512).
- **Krok patchy (stride, piksele)** (`STRIDE`)
  - Jak gęsto wycinać patch’e.
  - `STRIDE = PATCH_SIZE` → bez nakładania.
  - Mniejszy stride → więcej próbek (wolniej), często lepsze pokrycie.
- **Szerokość bazowa U-Net (base channels)** (`BASE_CHANNELS`)
  - Dotyczy głównie U-Net: większa wartość = większy model (wolniej, więcej RAM).
- **Liczba epok** (`EPOCHS`)
  - Więcej epok = dłuższy trening, potencjalnie lepszy wynik (do pewnego momentu).
- **Batch size** (`BATCH_SIZE`)
  - Zbyt duży batch może spowodować brak pamięci (szczególnie na GPU).
- **Learning rate** (`LEARNING_RATE`)
  - Zbyt duży: trening niestabilny; zbyt mały: trening bardzo wolny.
- **Udział walidacji (0.0 - 0.9)** (`VALIDATION_SPLIT`)
  - Ile danych odłożyć na walidację (typowo 0.2).
- **Losowe ziarno** (`RANDOM_SEED`)
  - Ułatwia powtarzalność wyników.
- **Użyj CUDA (jeśli dostępne)** (`USE_CUDA`)
  - `TAK`: używa GPU, jeśli PyTorch widzi CUDA.
- **Wyjściowy model (.pth)** (`OUTPUT_MODEL`)
  - Ścieżka pliku checkpointu `.pth`.

**Wyjście:**
- Plik `.pth` kompatybilny z algorytmem **Inferencja modelu (PyTorch / ONNX)**.
- Dodatkowo w wynikach Processing pojawia się tekstowy **Raport treningu**.

**Najczęstsze problemy:**
- Brak `torch` → zainstaluj PyTorch (strona pytorch.org).
- Wybrany model inny niż U-Net, a brak `segmentation-models-pytorch`/`timm` → doinstaluj te biblioteki.
- `PATCH_SIZE` niepodzielny przez 16 → ustaw np. 256.

---

## Inferencja modelu (PyTorch / ONNX)

**Do czego służy:** uruchamia wytrenowany model na całym rastrze wejściowym, kafelkując raster i sklejając wynik.

**Parametry (opcje):**
- **Wejściowy raster** (`INPUT`)
  - Raster, na którym chcesz policzyć predykcję.
- **Plik modelu (.pth, .pt lub .onnx)** (`MODEL_FILE`)
  - `.pth/.pt` → PyTorch
  - `.onnx` → ONNX
- **Format modelu** (`MODEL_FORMAT`)
  - `Auto` (domyślnie): wybiera po rozszerzeniu pliku.
  - `PyTorch`: wymaga `torch`.
  - `ONNX`: wymaga `onnxruntime`.
- **Rozmiar kafla (piksele)** (`TILE_SIZE`)
  - Typowo 256 lub 512.
- **Zakładka kafli (piksele)** (`OVERLAP`)
  - `0` (domyślnie) = bez zakładki.
  - `>0` = kafle nakładają się; wyniki są uśredniane na zakładkach (często lepsze na łączeniach).
  - Musi być **mniejsze niż rozmiar kafla**.
- **Liczba kanałów wejściowych** (`INPUT_CHANNELS`)
  - Ile pasm wejściowych podajesz do modelu.
- **Typ wyjścia** (`OUTPUT_TYPE`)
  - **Indeksy klas (argmax po pasmach, Int32)**: najlepsze dla wieloklasowej segmentacji.
  - **Sigmoid binary (próg 0.5, UInt8)**: do modeli 1-kanałowych (0/1).
  - **Softmax probability (Float32)**: mapa prawdopodobieństwa (najczęściej dla binarnej klasy, zależnie od modelu).
- **Wyjściowy raster** (`OUTPUT`)
  - Ścieżka rastra wynikowego.

**Najczęstsze problemy:**
- Wybrany format PyTorch, ale brak `torch` → zainstaluj PyTorch lub użyj modelu ONNX.
- Wybrany format ONNX, ale brak `onnxruntime` → doinstaluj `onnxruntime`.

---

# KLASYFIKACJA

## Klasyfikacja pikseli RGB (Euklidesowa / CIEDE2000)

**Do czego służy:** przypisuje każdemu pikselowi klasę na podstawie tego, do którego z podanych kolorów referencyjnych jest „najbliżej”.

**Parametry (opcje):**
- **Wejściowy raster (min. 3 pasma RGB)** (`INPUT`)
  - Raster musi mieć co najmniej 3 pasma.
- **Kolory modelowe (Nazwa, R, G, B)** (`COLORS`)
  - Tabelka: każdy wiersz to jedna klasa.
  - Kolumny:
    - `Nazwa`: dowolny opis (np. „Droga”).
    - `R`, `G`, `B`: wartości 0–255.
- **Metryka odległości** (`METRIC`)
  - **Euklidesowa (RGB)**: szybsza, zwykle wystarcza.
  - **CIEDE2000 (CIE Lab, D65)**: wolniejsza, ale bardziej „percepcyjna” (często lepsza przy podobnych kolorach).
- **Wyjściowy raster klas** (`OUTPUT`)
  - Raster wynikowy: wartości 0, 1, 2… (indeksy wierszy z tabeli kolorów).

**Wskazówki:**
- Zacznij od 3–6 klas i prostych, „czystych” kolorów referencyjnych.
- Jeśli dwie klasy mają bardzo podobny kolor, CIEDE2000 może dać stabilniejsze wyniki.

---

## SAGA: klasyfikatory i k-means (proxy)

To są pozycje:
- **SAGA: Sieć neuronowa**
- **SAGA: Drzewo decyzyjne**
- **SAGA: Regresja logistyczna**
- **SAGA: Normal Bayes**
- **SAGA: Random Forest**
- **SAGA: SVM**
- **SAGA: K-Means dla rastrów**

**Do czego służą:** są to „cienkie” mostki do algorytmów z providera **SAGA Next Gen** (`sagang:*`). Wtyczka Vectorization Bridge:
- pokazuje te algorytmy w swoim menu,
- a po uruchomieniu przekazuje parametry 1:1 do SAGA.

**Najważniejsza rzecz dla początkującego:**
- **Parametry w oknie tych algorytmów pochodzą z SAGA** (mogą się różnić między wersjami SAGA).
- W samym oknie Processing zwykle jest też opis (help) SAGA — warto go czytać.

**Wymaganie:**
- Musisz mieć zainstalowaną wtyczkę QGIS **„SAGA Next Gen”**.

---

# PRZETWARZANIE OBRAZU

## Wykrywanie krawędzi (Canny/Sobel/SAGA)

**Do czego służy:** tworzy raster krawędzi.

**Parametry (opcje):**
- **Wejściowy raster** (`INPUT`)
- **Metoda wykrywania krawędzi** (`METHOD`)
  - **Canny (cv2)**: cienkie krawędzie, najbardziej „klasyczne”.
  - **Sobel — gradient natężenia (cv2)**: mapa gradientu (często grubsze krawędzie).
  - **SAGA: ...**: pojawia się tylko, jeśli wykryto algorytm krawędzi w SAGA Next Gen.
- **Canny — próg dolny** (`CANNY_LOW`)
  - Zwiększ, gdy masz za dużo szumu (za dużo krawędzi).
- **Canny — próg górny** (`CANNY_HIGH`)
  - Zwiększ, gdy brakuje mocnych krawędzi.
  - Dla Canny musi być **większy niż próg dolny**.
- **Wyjściowy raster krawędzi** (`OUTPUT`)

**Wyjście:**
- Dla Canny: raster 0/255 (UInt8).
- Dla Sobel: raster Float32 (wartości gradientu).

**Najczęstsze problemy:**
- Brak `cv2` → doinstaluj `opencv-python`.
- Metoda SAGA nie jest widoczna → brak SAGA Next Gen lub SAGA nie udostępnia algorytmu krawędzi w rejestrze.

---

## Szkieletyzacja rastra (SAGA)

**Do czego służy:** deleguje do algorytmu SAGA `sagang:rasterskeletonization` (szkieletyzacja / „odchudzanie” obiektów do linii).

**Ważne:** parametry są dokładnie takie jak w SAGA (proxy).

---

## Seeded Region Growing

**Do czego służy:** segmentuje raster, „rozrastając” regiony od punktów startowych (seedów).

**Parametry (opcje):**
- **Wejściowy raster** (`INPUT`)
  - Dla RGB algorytm i tak liczy skalę szarości jako średnią z pasm.
- **Warstwa seedów (punkty)** (`SEED_LAYER`)
  - Warstwa punktowa; każdy punkt jest startem regionu.
- **Pole z identyfikatorem klasy (opcjonalne, integer)** (`CLASS_FIELD`)
  - Puste: każdy seed = osobny region (regiony numerowane 1..N).
  - Ustawione: seedy z tą samą wartością w polu tworzą jeden region.
- **Tolerancja (różnica intensywności od średniej regionu)** (`TOLERANCE`)
  - Kluczowy parametr.
  - Mała tolerancja → regiony rosną powoli i obejmują tylko bardzo podobne piksele.
  - Duża tolerancja → regiony mogą „zalać” dużą część rastra.
- **Wyjściowy raster regionów** (`OUTPUT`)

**Wyjście:**
- GeoTIFF Int32, `nodata = -1`.

**Wskazówki:**
- Zacznij od kilku seedów w „pewnych” miejscach.
- Jeśli region prawie nie rośnie, zwiększ `TOLERANCE`.
- Jeśli regiony pochłaniają za dużo, zmniejsz `TOLERANCE`.

---

## Rozdziel raster RGB na pasma

**Do czego służy:** rozbija raster (typowo RGB) na 3 osobne rastry jednopasmowe: **R**, **G**, **B**.

**Kiedy używać:** gdy algorytm (często SAGA) wymaga pojedynczego pasma na wejściu, a Ty masz raster RGB.

**Parametry (opcje):**
- **Wejściowy raster (min. 3 pasma)** (`INPUT`)
- **Pasmo R (czerwone)** (`OUTPUT_R`)
- **Pasmo G (zielone)** (`OUTPUT_G`)
- **Pasmo B (niebieskie)** (`OUTPUT_B`)

**Wyjście:**
- 3 GeoTIFF (LZW), z zachowaniem georeferencji i typu danych.

---

# Najczęstsze pytania

## Dlaczego część opcji (SAGA) nie działa?

Bo wymagana jest wtyczka QGIS **SAGA Next Gen**. Zainstaluj ją w:
`Plugins → Manage and Install Plugins → SAGA Next Gen`.

## Skąd mam wiedzieć, czego brakuje w moim QGIS?

Uruchom **Diagnostyka → Sprawdź zależności** — raport wprost powie, czy brakuje `cv2`, `torch`, `onnxruntime` lub SAGA.
