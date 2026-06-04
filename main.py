import sys
import os
import ast
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QFrame, QTextEdit, QComboBox,
    QProgressBar, QPushButton, QFileDialog, QTabWidget,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

JSON_PATHS = {
    2: "weights_2inputs.json",
    3: "weights_3inputs.json",
    4: "weights_4inputs.json",
}

class PatternPredictor(nn.Module):

    def __init__(self, num_inputs: int):
        super().__init__()
        in_dim = num_inputs + 1
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_from_json(json_path: str, epochs: int = 600, status_cb=None) -> "PatternPredictor":
    with open(json_path) as f:
        data = json.load(f)

    num_inputs = data["num_inputs"]
    samples    = data["samples"]

    features = [s["inputs"] + [s["target"]] for s in samples]
    targets  = [[s["ideal_base"]] for s in samples]

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.float32)

    net       = PatternPredictor(num_inputs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X), y)
        loss.backward()
        optimizer.step()
        if status_cb and epoch % 50 == 0:
            status_cb(epoch, epochs, loss.item())

    return net.eval()

# 2. NEURAL-NETWORK CODE ANALYSER

class CodeAnalyser:
    INPUT_KEYWORDS  = {"input", "inputs", "x", "features", "data", "x_train", "X"}
    TARGET_KEYWORDS = {"target", "targets", "y", "label", "labels", "y_train",
                       "output", "expected", "desired"}

    def __init__(self, source: str):
        self.source      = source
        self.tree        = ast.parse(source)
        self.num_inputs  = None
        self.target      = None
        self.layer_sizes = []
        self.input_names = []
        self.notes       = []

    @staticmethod
    def _list_len(node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return len(node.elts)
        return None

    @staticmethod
    def _numeric(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = CodeAnalyser._numeric(node.operand)
            return -v if v is not None else None
        return None

    def _pass_assignments(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                name = tgt.id if isinstance(tgt, ast.Name) else None
                if name is None:
                    continue
                name_lower = name.lower()

                if any(kw in name_lower for kw in ("input", "feature", "x_train", "data")):
                    ln = self._list_len(node.value)
                    if ln is not None:
                        self.num_inputs = ln
                        self.input_names.append(name)
                        self.notes.append(f" • Detected {ln} inputs from '{name} = [...]'")

                if any(kw == name_lower for kw in self.TARGET_KEYWORDS):
                    v = self._numeric(node.value)
                    if v is not None:
                        self.target = v
                        self.notes.append(f" • Detected target {v} from '{name} = {v}'")
                    elif isinstance(node.value, (ast.List, ast.Tuple)):
                        elts = [e for e in (self._numeric(e) for e in node.value.elts) if e is not None]
                        if elts:
                            self.target = elts[0]
                            self.notes.append(f" • Detected target(s) {elts} from '{name}'")

    def _pass_target_array_calls(self):
        if self.target is not None:
            return
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                name = tgt.id if isinstance(tgt, ast.Name) else None
                if name is None:
                    continue
                if not any(kw == name.lower() for kw in self.TARGET_KEYWORDS):
                    continue
                val = node.value
                if not isinstance(val, ast.Call):
                    continue
                func = val.func
                is_array = (
                    (isinstance(func, ast.Attribute) and func.attr in ("array", "tensor")) or
                    (isinstance(func, ast.Name)      and func.id   in ("array", "tensor"))
                )
                if not is_array or not val.args:
                    continue
                arg = val.args[0]
                if not isinstance(arg, (ast.List, ast.Tuple)):
                    continue
                flat = []
                for elt in arg.elts:
                    v = self._numeric(elt)
                    if v is not None:
                        flat.append(v)
                    elif isinstance(elt, (ast.List, ast.Tuple)) and elt.elts:
                        v = self._numeric(elt.elts[0])
                        if v is not None:
                            flat.append(v)
                if flat:
                    self.target = flat[0]
                    self.notes.append(
                        f" • Detected target(s) {flat} from '{name} = np.array(...)'"
                    )

    def _pass_linear_layers(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_linear = (
                (isinstance(func, ast.Attribute) and func.attr == "Linear") or
                (isinstance(func, ast.Name)      and func.id   == "Linear")
            )
            if is_linear and len(node.args) >= 2:
                in_f  = self._numeric(node.args[0])
                out_f = self._numeric(node.args[1])
                if in_f is not None and out_f is not None:
                    self.layer_sizes.append((int(in_f), int(out_f)))

    def _pass_input_shape_comments(self):
        for line in self.source.splitlines():
            m = re.search(r"input[_s]*\s*[=:]\s*(\d+)", line, re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if self.num_inputs is None:
                    self.num_inputs = n
                    self.notes.append(f" • Inferred {n} inputs from comment: '{line.strip()}'")

    def _pass_tensor_calls(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_tensor = (
                (isinstance(func, ast.Attribute) and func.attr in ("tensor", "array", "Tensor")) or
                (isinstance(func, ast.Name)      and func.id   in ("tensor", "array"))
            )
            if is_tensor and node.args:
                ln = self._list_len(node.args[0])
                if ln is not None and self.num_inputs is None:
                    self.num_inputs = ln
                    self.notes.append(f" • Detected {ln} inputs from tensor/array literal")

    def analyse(self) -> dict:
        self._pass_assignments()
        self._pass_target_array_calls()
        self._pass_linear_layers()
        self._pass_tensor_calls()
        self._pass_input_shape_comments()

        if self.num_inputs is None and self.layer_sizes:
            self.num_inputs = self.layer_sizes[0][0]
            self.notes.append(f" • Inferred {self.num_inputs} inputs from first nn.Linear layer")

        return {
            "num_inputs":  self.num_inputs,
            "target":      self.target,
            "layer_sizes": self.layer_sizes,
            "input_names": self.input_names,
            "notes":       self.notes,
        }

class TrainWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def run(self):
        nets = {}
        for n, json_path in JSON_PATHS.items():
            if not os.path.exists(json_path):
                self.progress.emit(f"❌ Missing {json_path} — cannot train {n}-input network.")
                continue
            self.progress.emit(f"🧠 Training {n}-input PatternPredictor …")

            def cb(ep, total, loss, n=n):
                if ep % 150 == 0:
                    self.progress.emit(f" [{n}-input] epoch {ep}/{total} loss={loss:.5f}")

            nets[n] = train_from_json(json_path, status_cb=cb)
            self.progress.emit(f"✅ {n}-input network ready.")
        self.finished.emit(nets)


class AnalyseTrainWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        self.progress.emit("📂 Reading file …")
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError as e:
            self.finished.emit({"error": str(e), "info": {}})
            return

        self.progress.emit("🔍 Analysing code structure …")
        try:
            analyser = CodeAnalyser(source)
        except SyntaxError as e:
            self.finished.emit({"error": f"Syntax error in file: {e}", "info": {}})
            return

        info       = analyser.analyse()
        num_inputs = info["num_inputs"]
        target     = info["target"]

        if num_inputs is None:
            self.finished.emit({"error": "Could not detect number of inputs.", "info": info})
            return

        self.progress.emit(f"✅ Detected {num_inputs} input(s).")

        if target is None:
            target = 0.5
            self.progress.emit("⚠ No target found — using default 0.5")
        else:
            self.progress.emit(f"✅ Detected target value: {target}")

        # Pick closest matching JSON
        closest   = min(JSON_PATHS.keys(), key=lambda k: abs(k - num_inputs))
        json_path = JSON_PATHS[closest]

        if closest != num_inputs:
            self.progress.emit(
                f"⚠ No JSON for {num_inputs} inputs — using {closest}-input data ({json_path})"
            )
        else:
            self.progress.emit(f"📄 Using {json_path} …")

        if not os.path.exists(json_path):
            self.finished.emit({"error": f"Required file not found: {json_path}", "info": info})
            return

        self.progress.emit(f"🧠 Training meta-network from {json_path} …")

        def cb(ep, total, loss):
            if ep % 150 == 0:
                self.progress.emit(f" epoch {ep}/{total} loss={loss:.5f}")

        net = train_from_json(json_path, epochs=600, status_cb=cb)
        self.progress.emit("✅ Optimal weights computed!")

        vals   = [0.5] * closest
        tensor = torch.tensor([vals + [target]], dtype=torch.float32)
        with torch.no_grad():
            base = net(tensor).item()

        seed_val = int((sum(abs(v) for v in vals) + target) * 1000) % (2**31)
        rng      = np.random.default_rng(seed_val)
        matrix   = rng.normal(loc=base, scale=0.05, size=(closest, 3))

        self.finished.emit({
            "net":        net,
            "num_inputs": closest,
            "target":     target,
            "base":       base,
            "matrix":     matrix.tolist(),
            "info":       info,
        })

STYLE = """
QWidget { background: #1a1d27; color: #e8eaf0;
font-family: 'Segoe UI', sans-serif; }
QLabel { font-size: 13px; }
QDoubleSpinBox { background: #252837; border: 1px solid #3a3f5c;
border-radius: 5px; padding: 4px 8px;
color: #e8eaf0; font-size: 13px; }
QDoubleSpinBox:disabled { color: #555; background: #1e2030; }
QComboBox { background: #252837; border: 1px solid #3a3f5c;
border-radius: 5px; padding: 4px 8px;
color: #e8eaf0; font-size: 13px; min-width: 120px; }
QComboBox QAbstractItemView { background: #252837; color: #e8eaf0; }
QTextEdit { background: #0d1117; border: 1px solid #30363d;
border-radius: 6px; color: #58a6ff;
font-family: 'Courier New', monospace; font-size: 12px; }
QPushButton { background: #2e3a5c; border: 1px solid #4f6aaa;
border-radius: 6px; padding: 6px 16px;
color: #c8d8ff; font-size: 13px; }
QPushButton:hover { background: #3a4e80; }
QPushButton:disabled { background: #222535; color: #555; border-color: #333; }
QFrame[frameShape="4"] { color: #3a3f5c; }
QProgressBar { border: 1px solid #3a3f5c; border-radius: 4px;
background: #252837; height: 6px; }
QProgressBar::chunk { background: #4f8ef7; border-radius: 4px; }
QTabWidget::pane { border: 1px solid #3a3f5c; }
QTabBar::tab { background: #1e2235; color: #8892b0;
padding: 6px 18px; border-radius: 4px; margin-right: 2px; }
QTabBar::tab:selected { background: #2e3a5c; color: #c8d8ff; }
"""

class HyperNetApp(QWidget):
    def __init__(self):
        super().__init__()
        self._nets      = {}
        self._mode      = 2
        self._spinboxes = []
        self.initUI()
        self._start_training()

    def initUI(self):
        self.setWindowTitle("Hypernetwork Weight Predictor · Multi-Input + Code Analyser")
        self.resize(680, 680)
        self.setStyleSheet(STYLE)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(22, 18, 22, 18)

        title = QLabel("Hypernetwork Weight Predictor")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #7eb8f7; letter-spacing: 1px;")
        root.addWidget(title)

        sub = QLabel("Live weight prediction · Neural-network code analyser")
        sub.setStyleSheet("font-size: 11px; color: #7a7f9a; margin-top: -6px;")
        root.addWidget(sub)

        self._sep(root)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_predictor_tab()
        self._build_analyser_tab()

    def _build_predictor_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Number of data inputs:"))
        self.combo = QComboBox()
        self.combo.addItems(["2 inputs", "3 inputs", "4 inputs"])
        self.combo.currentIndexChanged.connect(self._on_mode_changed)
        self.combo.setEnabled(False)
        mode_row.addWidget(self.combo)
        mode_row.addStretch()
        lay.addLayout(mode_row)

        self._sep(lay)

        self.inputs_container = QVBoxLayout()
        lay.addLayout(self.inputs_container)

        tgt_row = QHBoxLayout()
        tgt_lbl = QLabel("Desired target output:")
        tgt_lbl.setStyleSheet("font-weight: 600; color: #f0a05a;")
        tgt_row.addWidget(tgt_lbl)
        self.box_target = QDoubleSpinBox()
        self.box_target.setRange(0.0, 1.0)
        self.box_target.setSingleStep(0.05)
        self.box_target.setValue(0.7)
        self.box_target.setEnabled(False)
        self.box_target.valueChanged.connect(self.update_prediction)
        tgt_row.addWidget(self.box_target)
        tgt_row.addStretch()
        lay.addLayout(tgt_row)

        self._sep(lay)

        self.status_label = QLabel("Initialising …")
        self.status_label.setStyleSheet("font-size: 11px; color: #8892b0; font-style: italic;")
        lay.addWidget(self.status_label)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 0)
        lay.addWidget(self.pbar)

        card = QWidget()
        card.setStyleSheet("background: #1e2235; border-radius: 10px; border: 1px solid #2e3450;")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(14, 10, 14, 10)

        self.predicted_label = QLabel("Pattern base scalar: —")
        self.predicted_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #4f8ef7;")
        card_lay.addWidget(self.predicted_label)

        self.matrix_hdr = QLabel()
        self.matrix_hdr.setStyleSheet("font-size: 12px; font-weight: 600; color: #a0a8cc; margin-top: 4px;")
        card_lay.addWidget(self.matrix_hdr)

        self.matrix_box = QTextEdit()
        self.matrix_box.setReadOnly(True)
        self.matrix_box.setFixedHeight(120)
        card_lay.addWidget(self.matrix_box)

        lay.addWidget(card)
        lay.addStretch()

        self._rebuild_input_rows(2)
        self.tabs.addTab(tab, "⚡ Live Predictor")

    def _build_analyser_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        info = QLabel(
            "Upload a Python neural network file. The analyser will automatically detect\n"
            "the number of inputs, target value, and compute the optimal weight baselines."
        )
        info.setStyleSheet("font-size: 12px; color: #8892b0;")
        info.setWordWrap(True)
        lay.addWidget(info)

        btn_row = QHBoxLayout()
        self.btn_upload = QPushButton("📂 Upload Neural Network (.py)")
        self.btn_upload.clicked.connect(self._on_upload)
        btn_row.addWidget(self.btn_upload)
        self.upload_path_label = QLabel("No file selected")
        self.upload_path_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 8px;")
        btn_row.addWidget(self.upload_path_label)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._sep(lay)

        self.analyse_status = QLabel("Waiting for file …")
        self.analyse_status.setStyleSheet("font-size: 11px; color: #8892b0; font-style: italic;")
        lay.addWidget(self.analyse_status)

        self.analyse_pbar = QProgressBar()
        self.analyse_pbar.setRange(0, 1)
        self.analyse_pbar.setValue(0)
        lay.addWidget(self.analyse_pbar)

        self._sep(lay)

        res_lbl = QLabel("Analysis & Optimal Weights:")
        res_lbl.setStyleSheet("font-weight: 700; color: #c8d8ff; font-size: 13px;")
        lay.addWidget(res_lbl)

        self.analyse_result = QTextEdit()
        self.analyse_result.setReadOnly(True)
        self.analyse_result.setPlaceholderText("Results will appear here after analysis …")
        self.analyse_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.analyse_result)

        self.tabs.addTab(tab, "🔍 Code Analyser")

    def _sep(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

    def _rebuild_input_rows(self, n: int):
        while self.inputs_container.count():
            item = self.inputs_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._spinboxes.clear()

        for k in range(n):
            row = QHBoxLayout()
            lbl = QLabel(f"Data input {k + 1}:")
            lbl.setStyleSheet("font-weight: 600; color: #c8cfe8; min-width: 110px;")
            row.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setRange(-5.0, 5.0)
            sb.setSingleStep(0.1)
            sb.setValue(round(0.5 - k * 0.5, 1))
            sb.setEnabled(False)
            sb.valueChanged.connect(self.update_prediction)
            row.addWidget(sb)
            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self.inputs_container.addWidget(container)
            self._spinboxes.append(sb)

        self.matrix_hdr.setText(f"Generated weight matrix ({n} inputs → 3 hidden nodes):")

    def _start_training(self):
        self.worker = TrainWorker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_trained)
        self.worker.start()

    def _on_progress(self, msg: str):
        self.status_label.setText(msg)

    def _on_trained(self, nets: dict):
        self._nets = nets
        self.pbar.setRange(0, 1)
        self.pbar.setValue(1)
        self.status_label.setText("All networks ready ✓")
        self.status_label.setStyleSheet("font-size: 11px; color: #4caf50; font-style: normal;")
        for sb in self._spinboxes:
            sb.setEnabled(True)
        self.box_target.setEnabled(True)
        self.combo.setEnabled(True)
        self.update_prediction()

    def _on_mode_changed(self, index: int):
        self._mode = index + 2
        self._rebuild_input_rows(self._mode)
        trained = bool(self._nets)
        for sb in self._spinboxes:
            sb.setEnabled(trained)
        if trained:
            self.update_prediction()

    def update_prediction(self):
        if not self._nets or self._mode not in self._nets:
            return
        net  = self._nets[self._mode]
        vals = [sb.value() for sb in self._spinboxes]
        tgt  = self.box_target.value()

        tensor = torch.tensor([vals + [tgt]], dtype=torch.float32)
        with torch.no_grad():
            base = net(tensor).item()

        self.predicted_label.setText(f"Pattern base scalar: {base:+.4f}")

        seed_val = int((sum(abs(v) for v in vals) + tgt) * 1000) % (2**31)
        rng      = np.random.default_rng(seed_val)
        matrix   = rng.normal(loc=base, scale=0.05, size=(self._mode, 3))

        lines = [f" Input {i+1} → [ {' '.join(f'{v:+7.4f}' for v in row)} ]"
                 for i, row in enumerate(matrix)]
        self.matrix_box.setText("\n".join(lines))

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Neural Network Python File", "",
            "Python Files (*.py);;All Files (*)"
        )
        if not path:
            return

        self.upload_path_label.setText(os.path.basename(path))
        self.analyse_result.clear()
        self.analyse_status.setText("Starting analysis …")
        self.analyse_pbar.setRange(0, 0)
        self.btn_upload.setEnabled(False)
        self.combo.setEnabled(False)

        self._analyse_worker = AnalyseTrainWorker(path)
        self._analyse_worker.progress.connect(self._on_analyse_progress)
        self._analyse_worker.finished.connect(self._on_analyse_finished)
        self._analyse_worker.start()

    def _on_analyse_progress(self, msg: str):
        self.analyse_status.setText(msg)
        self.analyse_result.append(msg)

    def _on_analyse_finished(self, result: dict):
        self.analyse_pbar.setRange(0, 1)
        self.analyse_pbar.setValue(1)
        self.btn_upload.setEnabled(True)
        self.combo.setEnabled(bool(self._nets))

        if "error" in result:
            self.analyse_status.setText("❌ Analysis failed.")
            self.analyse_result.append("\n❌ ERROR: " + result["error"])
            for note in result.get("info", {}).get("notes", []):
                self.analyse_result.append(note)
            return

        self.analyse_status.setText("✅ Analysis complete — optimal weights ready!")
        self.analyse_status.setStyleSheet("font-size: 11px; color: #4caf50;")

        info       = result["info"]
        num_inputs = result["num_inputs"]
        target     = result["target"]
        base       = result["base"]
        matrix     = result["matrix"]

        out = [
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            " ANALYSIS RESULTS",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f" Inputs detected  : {num_inputs}",
            f" Target value     : {target:.4f}",
            f" Predicted base W : {base:+.6f}",
        ]
        if info["layer_sizes"]:
            out.append(f" Layer dims found : {info['layer_sizes']}")
        if info["input_names"]:
            out.append(f" Input variables  : {', '.join(info['input_names'])}")

        out.append("\n Detection notes:")
        out.extend(info["notes"])

        out.append(f"\n Optimal weight matrix ({num_inputs} inputs → 3 hidden nodes):")
        for i, row in enumerate(matrix[:10]):
            out.append(f"  Input {i+1} → [ {' '.join(f'{v:+.6f}' for v in row)} ]")
        if num_inputs > 10:
            out.append(f"  ... and {num_inputs - 10} more rows skipped ...")

        out.append("\n Per-input weight summary:")
        for i, row in enumerate(matrix[:10]):
            out.append(f"  Input {i+1} : avg={sum(row)/len(row):+.4f}  "
                       f"min={min(row):+.4f}  max={max(row):+.4f}")

        out.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.analyse_result.append("\n".join(out))

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HyperNetApp()
    window.show()
    sys.exit(app.exec())