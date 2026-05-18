"""
Algorytm trenowania modeli segmentacji/klasyfikacji pikseli w QGIS (PyTorch).

Wspierane architektury:
  • U-Net (własna implementacja)
  • DeepLabV3Plus / LinkNet / FPN / PSPNet / MAnet (segmentation_models_pytorch)
"""

import math

import numpy as np

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from ._torch_models import build_model
from ._compat import (
    torch_available, torch_version,
    smp_available, smp_version,
)


_QGIS_DTYPE_MAP = {
    1: np.uint8,
    2: np.uint16,
    3: np.int16,
    4: np.uint32,
    5: np.int32,
    6: np.float32,
    7: np.float64,
}


def _read_band(provider, band_index, w, h, extent):
    block = provider.block(band_index, extent, w, h)
    if not block.isValid():
        return np.zeros((h, w), dtype=np.float32)
    dtype = _QGIS_DTYPE_MAP.get(block.dataType(), np.float32)
    return np.frombuffer(bytes(block.data()), dtype=dtype).reshape(h, w).copy()


def _compute_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_image_patch(chw, x0, y0, patch_size):
    c, h, w = chw.shape
    x1 = min(x0 + patch_size, w)
    y1 = min(y0 + patch_size, h)
    patch = np.zeros((c, patch_size, patch_size), dtype=np.float32)
    patch[:, : y1 - y0, : x1 - x0] = chw[:, y0:y1, x0:x1]
    return patch


def _extract_mask_patch(mask, x0, y0, patch_size, ignore_index):
    h, w = mask.shape
    x1 = min(x0 + patch_size, w)
    y1 = min(y0 + patch_size, h)
    patch = np.full((patch_size, patch_size), ignore_index, dtype=np.int64)
    patch[: y1 - y0, : x1 - x0] = mask[y0:y1, x0:x1]
    return patch


class PytorchTrainingAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    MASK = 'MASK'
    INPUT_CHANNELS = 'INPUT_CHANNELS'
    NUM_CLASSES = 'NUM_CLASSES'
    MODEL_ARCH = 'MODEL_ARCH'
    ENCODER = 'ENCODER'
    PATCH_SIZE = 'PATCH_SIZE'
    STRIDE = 'STRIDE'
    BASE_CHANNELS = 'BASE_CHANNELS'
    EPOCHS = 'EPOCHS'
    BATCH_SIZE = 'BATCH_SIZE'
    LEARNING_RATE = 'LEARNING_RATE'
    VALIDATION_SPLIT = 'VALIDATION_SPLIT'
    RANDOM_SEED = 'RANDOM_SEED'
    USE_CUDA = 'USE_CUDA'
    OUTPUT_MODEL = 'OUTPUT_MODEL'
    REPORT = 'REPORT'

    MODEL_OPTIONS = [
        'U-Net',
        'DeepLabV3Plus',
        'LinkNet',
        'FPN',
        'PSPNet',
        'MAnet',
    ]
    MODEL_KEYS = [
        'unet',
        'deeplabv3plus',
        'linknet',
        'fpn',
        'pspnet',
        'manet',
    ]
    ENCODER_OPTIONS = [
        'resnet18',
        'resnet34',
        'resnet50',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'mobilenet_v3_large',
        'mobilenet_v3_small',
        'timm-gandalf',
        'timm-efficientnet-b0',
    ]

    def name(self):
        return 'pytorch_training'

    def displayName(self):
        return 'Trenowanie modelu (PyTorch)'

    def group(self):
        return 'Deep learning'

    def groupId(self):
        return 'deeplearning'

    def shortHelpString(self):
        tv = torch_version()
        sv = smp_version()
        t_status = f'✓ PyTorch {tv}' if tv else '✗ PyTorch niedostępny'
        s_status = f'✓ segmentation-models-pytorch {sv}' if sv else '✗ segmentation-models-pytorch niedostępny'
        return (
            'Trenuje model segmentacji pikselowej bezpośrednio w QGIS na bazie '
            'rastra wejściowego i rastra masek.\n\n'
            'Wybór modelu odbywa się przez listę rozwijaną "Model".\n'
            'Dla modeli SMP dostępna jest też lista "Encoder (backbone)".\n\n'
            'Maska powinna zawierać indeksy klas (0..N-1) w pierwszym paśmie.\n'
            'Piksele poza zakresem klas lub nodata są ignorowane podczas treningu.\n\n'
            'Wyjściem jest plik .pth z checkpointem (state_dict + metadane modelu), '
            'kompatybilny z algorytmem inferencji "Inferencja modelu (PyTorch / ONNX)".\n\n'
            f'Status: {t_status}; {s_status}'
        )

    def createInstance(self):
        return PytorchTrainingAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Raster wejściowy (obrazy treningowe)'
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.MASK, 'Raster masek (klasy w paśmie 1)'
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL_ARCH, 'Model', options=self.MODEL_OPTIONS, defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.ENCODER,
            'Encoder (backbone, dla modeli SMP)',
            options=self.ENCODER_OPTIONS,
            defaultValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.INPUT_CHANNELS, 'Liczba kanałów wejściowych',
            type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.NUM_CLASSES, 'Liczba klas',
            type=QgsProcessingParameterNumber.Integer, defaultValue=2, minValue=2
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.PATCH_SIZE, 'Rozmiar patcha (piksele)',
            type=QgsProcessingParameterNumber.Integer, defaultValue=256, minValue=32
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.STRIDE, 'Krok patchy (stride, piksele)',
            type=QgsProcessingParameterNumber.Integer, defaultValue=256, minValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BASE_CHANNELS, 'Szerokość bazowa U-Net (base channels)',
            type=QgsProcessingParameterNumber.Integer, defaultValue=32, minValue=8
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.EPOCHS, 'Liczba epok',
            type=QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BATCH_SIZE, 'Batch size',
            type=QgsProcessingParameterNumber.Integer, defaultValue=8, minValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.LEARNING_RATE, 'Learning rate',
            type=QgsProcessingParameterNumber.Double, defaultValue=0.001, minValue=1e-8
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.VALIDATION_SPLIT, 'Udział walidacji (0.0 - 0.9)',
            type=QgsProcessingParameterNumber.Double, defaultValue=0.2, minValue=0.0, maxValue=0.9
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RANDOM_SEED, 'Losowe ziarno',
            type=QgsProcessingParameterNumber.Integer, defaultValue=42, minValue=0
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_CUDA, 'Użyj CUDA (jeśli dostępne)', defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_MODEL, 'Wyjściowy model (.pth)',
            fileFilter='Model PyTorch (*.pth)'
        ))

        self.addOutput(QgsProcessingOutputString(
            self.REPORT, 'Raport treningu'
        ))

    def checkParameterValues(self, parameters, context):
        patch_size = self.parameterAsInt(parameters, self.PATCH_SIZE, context)
        stride = self.parameterAsInt(parameters, self.STRIDE, context)
        model_arch = self.parameterAsEnum(parameters, self.MODEL_ARCH, context)
        if patch_size % 16 != 0:
            return False, 'Rozmiar patcha musi być podzielny przez 16.'
        if stride < 1:
            return False, 'Stride musi być większy lub równy 1.'
        if not torch_available():
            return False, (
                'Brak biblioteki torch (PyTorch). '
                'Zainstaluj PyTorch: https://pytorch.org/get-started/locally/'
            )
        if model_arch != 0 and not smp_available():
            return False, (
                'Wybrana architektura wymaga segmentation_models_pytorch. '
                'Zainstaluj: pip install segmentation-models-pytorch timm'
            )
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        in_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        mask_layer = self.parameterAsRasterLayer(parameters, self.MASK, context)
        model_arch = self.parameterAsEnum(parameters, self.MODEL_ARCH, context)
        encoder_idx = self.parameterAsEnum(parameters, self.ENCODER, context)
        in_channels = self.parameterAsInt(parameters, self.INPUT_CHANNELS, context)
        num_classes = self.parameterAsInt(parameters, self.NUM_CLASSES, context)
        patch_size = self.parameterAsInt(parameters, self.PATCH_SIZE, context)
        stride = self.parameterAsInt(parameters, self.STRIDE, context)
        base_channels = self.parameterAsInt(parameters, self.BASE_CHANNELS, context)
        epochs = self.parameterAsInt(parameters, self.EPOCHS, context)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        learning_rate = self.parameterAsDouble(parameters, self.LEARNING_RATE, context)
        val_split = self.parameterAsDouble(parameters, self.VALIDATION_SPLIT, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        use_cuda = self.parameterAsBool(parameters, self.USE_CUDA, context)
        output_model_path = self.parameterAsString(parameters, self.OUTPUT_MODEL, context)
        model_key = self.MODEL_KEYS[model_arch]
        encoder_name = self.ENCODER_OPTIONS[encoder_idx]

        in_provider = in_layer.dataProvider()
        mask_provider = mask_layer.dataProvider()

        in_w = in_provider.xSize()
        in_h = in_provider.ySize()
        mask_w = mask_provider.xSize()
        mask_h = mask_provider.ySize()
        if in_w != mask_w or in_h != mask_h:
            feedback.reportError(
                'Raster wejściowy i maska muszą mieć tę samą rozdzielczość (width/height).',
                fatalError=True,
            )
            return {}
        if in_channels > in_provider.bandCount():
            feedback.reportError(
                f'Raster wejściowy ma {in_provider.bandCount()} pasm, '
                f'ale INPUT_CHANNELS={in_channels}.',
                fatalError=True,
            )
            return {}

        in_extent = in_layer.extent()
        mask_extent = mask_layer.extent()
        eps = 1e-9
        if (
            abs(in_extent.xMinimum() - mask_extent.xMinimum()) > eps
            or abs(in_extent.xMaximum() - mask_extent.xMaximum()) > eps
            or abs(in_extent.yMinimum() - mask_extent.yMinimum()) > eps
            or abs(in_extent.yMaximum() - mask_extent.yMaximum()) > eps
        ):
            feedback.reportError(
                'Raster wejściowy i maska muszą mieć ten sam extent.',
                fatalError=True,
            )
            return {}

        feedback.pushInfo(f'PyTorch: {torch.__version__}')
        device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        feedback.pushInfo(f'Urządzenie treningowe: {device}')
        feedback.pushInfo(f'Raster: {in_w}x{in_h}, kanały={in_channels}, klasy={num_classes}')
        feedback.pushInfo(
            f'Model: {self.MODEL_OPTIONS[model_arch]}'
            + (f', encoder={encoder_name}' if model_key != 'unet' else '')
        )

        image_bands = []
        for band_idx in range(1, in_channels + 1):
            image_bands.append(
                _read_band(in_provider, band_idx, in_w, in_h, in_extent).astype(np.float32)
            )
        image_chw = np.stack(image_bands, axis=0)

        mask_raw = _read_band(mask_provider, 1, in_w, in_h, mask_extent)
        mask_i64 = np.rint(mask_raw).astype(np.int64)
        ignore_index = -1

        if mask_provider.sourceHasNoDataValue(1):
            nodata = mask_provider.sourceNoDataValue(1)
            mask_i64[np.isclose(mask_raw, nodata)] = ignore_index

        valid_range = (mask_i64 >= 0) & (mask_i64 < num_classes)
        mask_i64[~valid_range] = ignore_index
        valid_pixels = int(np.count_nonzero(mask_i64 != ignore_index))
        if valid_pixels == 0:
            feedback.reportError(
                'Maska nie zawiera poprawnych pikseli klas (0..NUM_CLASSES-1).',
                fatalError=True,
            )
            return {}

        x_starts = _compute_starts(in_w, patch_size, stride)
        y_starts = _compute_starts(in_h, patch_size, stride)
        total_tiles = len(x_starts) * len(y_starts)

        tiles_x = []
        tiles_y = []
        for y0 in y_starts:
            for x0 in x_starts:
                if feedback.isCanceled():
                    return {}
                x_patch = _extract_image_patch(image_chw, x0, y0, patch_size)
                y_patch = _extract_mask_patch(mask_i64, x0, y0, patch_size, ignore_index)
                if np.count_nonzero(y_patch != ignore_index) == 0:
                    continue
                tiles_x.append(x_patch / 255.0)
                tiles_y.append(y_patch)

        n_samples = len(tiles_x)
        if n_samples < 2:
            feedback.reportError(
                f'Za mało patchy treningowych po filtracji: {n_samples} (minimum 2).',
                fatalError=True,
            )
            return {}

        x_np = np.stack(tiles_x, axis=0).astype(np.float32)
        y_np = np.stack(tiles_y, axis=0).astype(np.int64)
        feedback.pushInfo(
            f'Patchy: {n_samples} (z {total_tiles}); shape={x_np.shape}; valid_pixels={valid_pixels}'
        )

        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        val_count = int(math.floor(n_samples * val_split))
        if val_count >= n_samples:
            val_count = n_samples - 1
        train_count = n_samples - val_count
        train_idx = indices[:train_count]
        val_idx = indices[train_count:]

        x_train = torch.from_numpy(x_np[train_idx])
        y_train = torch.from_numpy(y_np[train_idx])
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        if val_count > 0:
            x_val = torch.from_numpy(x_np[val_idx])
            y_val = torch.from_numpy(y_np[val_idx])
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None

        try:
            model = build_model(
                torch,
                model_type=model_key,
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                encoder_name=encoder_name,
            )
        except Exception as e:
            feedback.reportError(f'Nie udało się zbudować modelu: {e}', fatalError=True)
            return {}
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        report_lines = [
            'Vectorization Bridge — raport treningu PyTorch',
            f'Model: {self.MODEL_OPTIONS[model_arch]}',
            f'Model key: {model_key}',
            f'Encoder: {encoder_name}' if model_key != 'unet' else 'Encoder: n/a (U-Net)',
            f'Kanały wejściowe: {in_channels}',
            f'Liczba klas: {num_classes}',
            f'Patch size: {patch_size}',
            f'Stride: {stride}',
            f'Samples: {n_samples} (train={train_count}, val={val_count})',
            f'Urządzenie: {device}',
            f'Epoki: {epochs}, batch_size={batch_size}, lr={learning_rate}',
        ]

        for epoch in range(1, epochs + 1):
            if feedback.isCanceled():
                return {}

            model.train()
            train_loss_sum = 0.0
            train_steps = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_steps += 1

            train_loss = train_loss_sum / max(train_steps, 1)

            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_steps = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss_sum += float(loss.item())
                        val_steps += 1
                val_loss = val_loss_sum / max(val_steps, 1)
                msg = f'Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}'
            else:
                val_loss = float('nan')
                msg = f'Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}'

            report_lines.append(msg)
            feedback.pushInfo(msg)
            feedback.setProgress(int(epoch * 100 / epochs))

        model = model.to('cpu')
        model.eval()
        checkpoint = {
            'format': 'vectorization_bridge_torch_checkpoint_v1',
            'model_type': model_key,
            'encoder_name': encoder_name if model_key != 'unet' else None,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'base_channels': base_channels,
            'state_dict': model.state_dict(),
        }
        torch.save(checkpoint, output_model_path)
        feedback.pushInfo(f'Model zapisany: {output_model_path}')
        report_lines.append(f'Output model: {output_model_path}')

        report = '\n'.join(report_lines)
        return {
            self.OUTPUT_MODEL: output_model_path,
            self.REPORT: report,
        }
