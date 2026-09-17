"""PyQt desktop UI for DeepDream.

Wires the same building blocks cli.py uses (deepdream.core, .image_utils,
.models) into a window a non-technical user can click through instead of
the command line. The dream/gradient-ascent code itself is untouched --
this file only adds a UI on top.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

# When PyInstaller-frozen, model weights are bundled under a "torch_home"
# folder alongside the executable instead of ~/.cache/torch -- point torch's
# hub cache at it before anything imports torch, so get_model() finds them
# offline instead of trying to download. No-op for `python gui.py` (dev run).
if getattr(sys, "frozen", False):
    os.environ.setdefault("TORCH_HOME", os.path.join(sys._MEIPASS, "torch_home"))

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepdream.core import run_octaves
from deepdream.image_utils import ArrayHWC, build_octave_sizes, load_image, save_image
from deepdream.models import AVAILABLE_MODELS, get_model, list_layers, make_guided_grad_fn, make_torch_grad_fn

# Plain-language explanations shown via the "(i)" button next to each control.
# Keep these free of ML jargon -- the audience is a non-technical friend, not
# a fellow engineer. Keyed by the same string used as the QFormLayout row label.
PARAM_HELP = {
    "Source image": "The photo you want to turn into a dream.",
    
    "Guide image": "An optional second picture. Instead of hallucinating generic shapes from within the model"
    "the dream will try to pull in textures and forms from this guide image instead. Choose a small model"
    "for this task (mobilenet / efficientnet / resnet)",
    
    "Model": "Which pretrained AI model looks at your image. All models here are trained on a dataset of animals/dogs mainly "
    "and so will dream as such. Other models can be found, trained to classify different objects. If unsure, leave it the default."
    "The default is inception_v3, but you can go for a slightly smaller models (mobilenet_v2, mobilenet_v3_large, efficientnet_b0),"
    "the medium sized resnet50, and the slightly larger (so, more expensive) vgg19 too",
    
    "Layer": "How deep into the AI's \"visual cortex\" the dream effect is applied. Earlier layers tend to "
    "produce swirly textures and edges; deeper layers produce more recognizable shapes and objects.",
    
    "Channel": "Narrows the effect to one specific pattern the AI knows (say, a dog's face as opposed to a cat's)"
    "instead of all of them at once. -1 means \"use everything\" (the usual choice). Only change this if you're experimenting.",
    
    "Octaves": "How many times the image is processed at different zoom levels, from zoomed-out to "
    "zoomed-in. More octaves add detail at multiple scales, but takes longer to run.",
    
    "Octave scale": "How much bigger each zoom level is than the last. Bigger numbers mean a bigger jump "
    "in scale between octaves, which can make the effect more dramatic but less smooth.",
    
    "Iterations per octave": "How many times the dream effect is applied at each zoom level. More "
    "iterations means a stronger, more intense effect, but also more processing time.",
    
    "Step size": "How big a nudge is applied to the image on each iteration. Bigger steps make the "
    "effect stronger and faster to appear, but can also make it look harsh, saturated, and/or noisy.",
    
    "Jitter": "Shifts the image by a few random pixels each iteration before processing. This avoids "
    "tile-like repeating patterns and makes the result look more natural.",
    
    "Max input size": "The image is shrunk to at most this many pixels (on its longest side) before "
    "dreaming, then the result is produced at that size. Smaller values run faster; larger values keep "
    "give higher resolution but take longer and use more memory,",
}


def resource_path(relative_path: str) -> str:
    """Resolve a bundled resource, whether running from source or a PyInstaller build."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


def array_to_pixmap(image_hwc: ArrayHWC) -> QPixmap:
    """Convert this project's float32 [0,1] HWC array format into a QPixmap."""
    clipped = np.clip(image_hwc, 0.0, 1.0)
    rgb = np.ascontiguousarray((clipped * 255.0).astype(np.uint8))
    h, w, _ = rgb.shape
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # .copy() so the QImage owns its pixel data once `rgb` goes out of scope.
    return QPixmap.fromImage(qimage.copy())


class StreamToSignal:
    """File-like object so the dream loop's progress prints (core.py calls
    plain print()) land in the GUI's log panel instead of a terminal."""

    def __init__(self, emit_fn):
        self._emit = emit_fn

    def write(self, text: str) -> None:
        if text:
            self._emit(text)

    def flush(self) -> None:
        pass


class DreamWorker(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(np.ndarray)
    failed = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = StreamToSignal(self.log.emit)
        try:
            p = self.params
            model = get_model(p["model"])
            layer = p["layer"] or AVAILABLE_MODELS[p["model"]].default_layer

            original = load_image(p["image"], max_size=p["max_size"])
            sizes = build_octave_sizes(original.shape[:2], p["octaves"], p["octave_scale"])

            min_px = AVAILABLE_MODELS[p["model"]].min_recommended_input_px
            if min(sizes[0]) < min_px:
                print(
                    f"Warning: smallest octave {sizes[0]} is below the recommended "
                    f"{min_px}px for {p['model']} -- consider fewer octaves or a "
                    "smaller octave scale.\n"
                )

            if p["guide"]:
                guide_image = load_image(p["guide"], max_size=p["max_size"])
                grad_fn = make_guided_grad_fn(model, layer, guide_image)
            else:
                grad_fn = make_torch_grad_fn(model, layer, p["channel"])

            result = run_octaves(original, sizes, p["iterations"], grad_fn, p["step_size"], p["jitter"])
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            sys.stdout = old_stdout


class LayerListWorker(QThread):
    """Loads a model off the GUI thread just to enumerate its layer names
    (some models take a few seconds to construct/load weights)."""

    loaded = pyqtSignal(str, list)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self) -> None:
        model = get_model(self.model_name)
        self.loaded.emit(self.model_name, list_layers(model))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeroDream")
        self.worker: Optional[DreamWorker] = None
        self.result_image: Optional[ArrayHWC] = None
        self.layer_worker: Optional[LayerListWorker] = None
        self._layer_cache: dict[str, list[str]] = {}

        self.image_path = QLineEdit()
        self.guide_path = QLineEdit()

        self.model_box = QComboBox()
        self.model_box.addItems(list(AVAILABLE_MODELS))
        self.model_box.currentTextChanged.connect(self._update_layer_placeholder)

        self.layer_box = QComboBox()
        self.layer_box.setEditable(True)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(-1, 4096)
        self.channel_spin.setValue(-1)

        self.octaves_spin = QSpinBox()
        self.octaves_spin.setRange(1, 10)
        self.octaves_spin.setValue(4)

        self.octave_scale_spin = QDoubleSpinBox()
        self.octave_scale_spin.setRange(1.01, 3.0)
        self.octave_scale_spin.setSingleStep(0.1)
        self.octave_scale_spin.setValue(1.8)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 500)
        self.iterations_spin.setValue(15)

        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setDecimals(4)
        self.step_size_spin.setRange(0.0001, 1.0)
        self.step_size_spin.setSingleStep(0.001)
        self.step_size_spin.setValue(0.005)

        self.jitter_spin = QSpinBox()
        self.jitter_spin.setRange(0, 64)
        self.jitter_spin.setValue(16)

        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(64, 4096)
        self.max_size_spin.setValue(768)

        self.preview_label = QLabel("No result yet")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 360)
        self.preview_label.setStyleSheet("border: 1px solid gray;")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        self.dream_button = QPushButton("Dream")
        self.dream_button.clicked.connect(self._start_dream)
        self.save_button = QPushButton("Save Result As...")
        self.save_button.clicked.connect(self._save_result)
        self.save_button.setEnabled(False)

        self._update_layer_placeholder(self.model_box.currentText())
        self._build_layout()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Source image", self._with_info(self._path_row(self.image_path, self._browse_image), "Source image"))
        form.addRow("Guide image (optional)", self._with_info(self._path_row(self.guide_path, self._browse_guide), "Guide image"))
        form.addRow("Model", self._with_info(self.model_box, "Model"))
        form.addRow("Layer (blank = model default)", self._with_info(self.layer_box, "Layer"))
        form.addRow("Channel (-1 = whole layer)", self._with_info(self.channel_spin, "Channel"))
        form.addRow("Octaves", self._with_info(self.octaves_spin, "Octaves"))
        form.addRow("Octave scale", self._with_info(self.octave_scale_spin, "Octave scale"))
        form.addRow("Iterations per octave", self._with_info(self.iterations_spin, "Iterations per octave"))
        form.addRow("Step size", self._with_info(self.step_size_spin, "Step size"))
        form.addRow("Jitter (px)", self._with_info(self.jitter_spin, "Jitter"))
        form.addRow("Max input size (px)", self._with_info(self.max_size_spin, "Max input size"))

        controls = QGroupBox("Settings")
        controls.setLayout(form)

        buttons = QHBoxLayout()
        buttons.addWidget(self.dream_button)
        buttons.addWidget(self.save_button)

        left = QVBoxLayout()
        left.addWidget(controls)
        left.addLayout(buttons)
        left.addWidget(QLabel("Log"))
        left.addWidget(self.log_view)

        right = QVBoxLayout()
        right.addWidget(self.preview_label)

        root = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(left_widget, 1)
        root.addWidget(right_widget, 1)

        outer = QVBoxLayout()
        outer.addWidget(self._build_logo_banner())
        outer.addLayout(root)

        central = QWidget()
        central.setLayout(outer)
        self.setCentralWidget(central)

    def _build_logo_banner(self) -> QWidget:
        """Full-width header banner showing the project logo/wordmark."""
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(resource_path(os.path.join("images", "logo.jpg")))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            logo_label.setText("NeroDream")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(logo_label)

        banner = QWidget()
        banner.setLayout(layout)
        banner.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return banner

    def _path_row(self, line_edit: QLineEdit, browse_slot) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(line_edit)
        button = QPushButton("Browse...")
        button.clicked.connect(browse_slot)
        row.addWidget(button)
        widget = QWidget()
        widget.setLayout(row)
        return widget

    def _with_info(self, widget: QWidget, help_key: str) -> QWidget:
        """Wrap a control with a small "(i)" button that shows a plain-language
        explanation on hover/click -- so non-technical users aren't stuck
        guessing what a slider or field does."""
        info_button = QToolButton()
        info_button.setText("ⓘ")  # circled small "i"
        info_button.setAutoRaise(True)
        info_button.setToolTip(PARAM_HELP.get(help_key, ""))
        info_button.setCursor(Qt.CursorShape.WhatsThisCursor)
        info_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Also show it on click for users who don't discover hover tooltips.
        info_button.clicked.connect(
            lambda: QMessageBox.information(self, help_key, PARAM_HELP.get(help_key, ""))
        )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        row.addWidget(info_button)
        container = QWidget()
        container.setLayout(row)
        return container

    def _update_layer_placeholder(self, model_name: str) -> None:
        self.layer_box.clearEditText()
        self.layer_box.lineEdit().setPlaceholderText(AVAILABLE_MODELS[model_name].default_layer)

        cached = self._layer_cache.get(model_name)
        if cached is not None:
            self._populate_layer_box(cached)
            return

        self.layer_box.clear()
        if self.layer_worker is not None and self.layer_worker.isRunning():
            self.layer_worker.loaded.disconnect(self._on_layers_loaded)
        self.layer_worker = LayerListWorker(model_name)
        self.layer_worker.loaded.connect(self._on_layers_loaded)
        self.layer_worker.start()

    def _on_layers_loaded(self, model_name: str, layers: list) -> None:
        self._layer_cache[model_name] = layers
        if self.model_box.currentText() == model_name:
            self._populate_layer_box(layers)

    def _populate_layer_box(self, layers: list) -> None:
        current_text = self.layer_box.currentText()
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        self.layer_box.addItems(layers)
        self.layer_box.setCurrentText(current_text)
        self.layer_box.blockSignals(False)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose source image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path.setText(path)

    def _browse_guide(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose guide image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.guide_path.setText(path)

    def _start_dream(self) -> None:
        if not self.image_path.text():
            QMessageBox.warning(self, "Missing image", "Pick a source image first.")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        params = {
            "image": self.image_path.text(),
            "guide": self.guide_path.text() or None,
            "model": self.model_box.currentText(),
            "layer": self.layer_box.currentText().strip(),
            "channel": self.channel_spin.value(),
            "octaves": self.octaves_spin.value(),
            "octave_scale": self.octave_scale_spin.value(),
            "iterations": self.iterations_spin.value(),
            "step_size": self.step_size_spin.value(),
            "jitter": self.jitter_spin.value(),
            "max_size": self.max_size_spin.value(),
        }

        self.log_view.clear()
        self.dream_button.setEnabled(False)
        self.dream_button.setText("Dreaming...")
        self.save_button.setEnabled(False)

        self.worker = DreamWorker(params)
        self.worker.log.connect(self._append_log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        self.log_view.insertPlainText(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _on_finished(self, result: np.ndarray) -> None:
        self.result_image = result
        self.preview_label.setPixmap(
            array_to_pixmap(result).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.dream_button.setEnabled(True)
        self.dream_button.setText("Dream")
        self.save_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.dream_button.setEnabled(True)
        self.dream_button.setText("Dream")
        QMessageBox.critical(self, "Dream failed", message)

    def _save_result(self) -> None:
        if self.result_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result", "dream.jpg", "JPEG (*.jpg)")
        if path:
            save_image(self.result_image, path)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1050, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
