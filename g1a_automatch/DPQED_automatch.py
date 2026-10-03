#!/usr/bin/env python3
"""
DPQED_automatch -- GUI for automatic geolocation assessment (G1A / NISAR vs
an L8_ref / C1 / any RIVAL-readable reference collection).

Every input of the matcher is a control here: input image and channels,
reference folder and band, detector + matcher, window sizes, keypoints,
large-offset search, RANSAC and consensus. 'Run' writes a job file and runs
automatch_job.py in a separate process, so the window stays responsive, a GPU
failure cannot take the GUI down, and 'Stop' frees the GPU at once. The same
job file runs unchanged from the command line:

    python automatch_job.py run <output_dir>/job_gui.json

Results: one RIVAL CSV per channel x detector x sweep point (open in
DPQED_rival.py with 'Load CSV'), RIVAL_BEST_<scene>_<channel>.csv, and
RUN_MANIFEST.csv. Double-click a result row to open its folder.

Runs with PyQt5, PyQt6 or PySide6 -- whichever is installed.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    QT_API = 'PyQt5'
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        QT_API = 'PyQt6'
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets
        QT_API = 'PySide6'

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_SCRIPT = os.path.join(HERE, 'automatch_job.py')
sys.path.insert(0, HERE)
from automatch_job import DEFAULT_JOB, PROGRESS_PREFIX  # noqa: E402  (no torch import)

Qt = QtCore.Qt
CHECKED = Qt.CheckState.Checked if QT_API != 'PyQt5' else Qt.Checked
UNCHECKED = Qt.CheckState.Unchecked if QT_API != 'PyQt5' else Qt.Unchecked
USER_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable if QT_API != 'PyQt5' else Qt.ItemIsUserCheckable
ITEM_ENABLED = Qt.ItemFlag.ItemIsEnabled if QT_API != 'PyQt5' else Qt.ItemIsEnabled
SEL_ROWS = (QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows if QT_API != 'PyQt5'
            else QtWidgets.QAbstractItemView.SelectRows)
NO_EDIT = (QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers if QT_API != 'PyQt5'
           else QtWidgets.QAbstractItemView.NoEditTriggers)
PROC_NOT_RUNNING = (QtCore.QProcess.ProcessState.NotRunning if QT_API != 'PyQt5'
                else QtCore.QProcess.NotRunning)
MERGED = (QtCore.QProcess.ProcessChannelMode.MergedChannels if QT_API != 'PyQt5'
          else QtCore.QProcess.MergedChannels)
MSG_YES = (QtWidgets.QMessageBox.StandardButton.Yes if QT_API != 'PyQt5'
           else QtWidgets.QMessageBox.Yes)
_DBB = QtWidgets.QDialogButtonBox
BTN_OK = _DBB.StandardButton.Ok if QT_API != 'PyQt5' else _DBB.Ok
BTN_CANCEL = _DBB.StandardButton.Cancel if QT_API != 'PyQt5' else _DBB.Cancel
BTN_DEFAULTS = _DBB.StandardButton.RestoreDefaults if QT_API != 'PyQt5' else _DBB.RestoreDefaults

RANSAC_CHOICES = ['magsac', 'ransac', 'lmeds', 'accurate']
COARSE_CHOICES = ['auto', 'matcher', 'phasecorr', 'manual', 'none']
REF_MODES = ['auto', 'index-shp', 'sidecar', 'degree-tile']
RESULT_COLS = ['sweep', 'channel', 'detector', 'status', 'truth_rmse_m', 'n_points', 'n_chips', 'mean_dx_m',
               'mean_dy_m', 'dE_min_m', 'dE_max_m', 'dN_min_m', 'dN_max_m', 'affine_rot_deg',
               'affine_resid_rmse_m', 'rmse_x_m', 'rmse_y_m', 'ce90_m', 'rival_csv']
CONSENSUS_MODELS = ['surface', 'constant', 'none']
SURFACE_DEGREES = ['auto', 'affine', 'bilinear', 'quadratic', 'biquadratic']
MAX_LOG_LINES = 5000


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_float_list(text, name):
    vals = [t for t in re.split(r'[,\s;]+', text.strip()) if t]
    try:
        return [float(v) for v in vals]
    except ValueError:
        raise ValueError(f'{name}: expected numbers separated by commas, got {text!r}')


def parse_int_list(text, name, allow_auto=False):
    out = []
    for t in [t for t in re.split(r'[,\s;]+', text.strip()) if t]:
        if allow_auto and t.lower() == 'auto':
            out.append(None)
            continue
        try:
            out.append(int(float(t)))
        except ValueError:
            raise ValueError(f'{name}: expected whole numbers separated by commas, got {text!r}')
    return out


def parse_band_map(text):
    """'band1:4, band2:3' -> {'band1': 4, 'band2': 3}"""
    out = {}
    for tok in [t for t in re.split(r'[,;]+', text.strip()) if t.strip()]:
        if ':' not in tok:
            raise ValueError(f'band map: expected channel:band, got {tok!r}')
        k, v = tok.split(':', 1)
        out[k.strip()] = int(v)
    return out


def fmt_list(vals):
    return ', '.join('auto' if v is None else (f'{v:g}' if isinstance(v, float) else str(v))
                     for v in vals)


class PathRow(QtWidgets.QWidget):
    """Line edit + browse button(s)."""

    def __init__(self, kind='dir', filt='All files (*)', parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        lay.addWidget(self.edit, 1)
        self.kind, self.filt = kind, filt
        if kind in ('file', 'file_or_dir'):
            b = QtWidgets.QPushButton('File…')
            b.clicked.connect(self._pick_file)
            lay.addWidget(b)
        if kind in ('dir', 'file_or_dir'):
            b = QtWidgets.QPushButton('Folder…')
            b.clicked.connect(self._pick_dir)
            lay.addWidget(b)

    def _pick_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select file', self.text(), self.filt)
        if p:
            self.edit.setText(p)

    def _pick_dir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder', self.text())
        if p:
            self.edit.setText(p)

    def text(self):
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t or '')


GROUP_TITLES = {
    'detector': 'Detector',
    'lgm': "LightGlue matcher — used when 'lgm' is ticked",
    'ada': "AdaLAM matcher — used when 'ada' is ticked",
    'snn': "SNN matcher (ratio test) — used when 'snn' is ticked",
    'fginn': "FGINN matcher — used when 'fginn' is ticked",
    'api': 'imcui thresholds',
    'feature': 'imcui detector model',
    'matcher': 'imcui matcher model',
}


def _group_of(spec):
    if spec.get('scope', 'detector') != 'detector':
        return spec['scope']
    head = spec['name'].split('.', 1)[0]
    return head if '.' in spec['name'] and head in GROUP_TITLES else 'detector'


def fmt_value(v):
    if isinstance(v, bool):
        return 'on' if v else 'off'
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


def prune_params(specs, values):
    """Drop parameters left at [default]; what remains goes into the job."""
    defaults = {sp['name']: sp['default'] for sp in specs}
    return {k: v for k, v in values.items() if k in defaults and list(v) != [defaults[k]]}


def count_variants(specs, values):
    """(detector variants, {matcher scope: passes}) for the dialog/summary."""
    n, passes = 1, {}
    for sp in specs:
        k = len(values.get(sp['name']) or [sp['default']])
        if sp.get('scope', 'detector') == 'detector':
            n *= k
        else:
            passes[sp['scope']] = passes.get(sp['scope'], 1) * k
    return n, passes


class ParamDialog(QtWidgets.QDialog):
    """Parameters of one detector (and of the matchers it uses).

    Choices and on/off settings are ticked, numbers and text are typed as a
    comma-separated list. Several values = try each one: every combination of
    detector values runs as its own variant and competes for RIVAL_BEST;
    matcher values add matching passes to each variant."""

    def __init__(self, name, specs, values=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{name} — parameters')
        self.name = name
        self.specs = list(specs)
        self.widgets = {}
        values = dict(values or {})
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            'Tick, or list comma-separated, every value to try. Each combination of '
            'detector values runs as its own variant and competes for the best result; '
            'matcher values add matching passes.')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        forms = {}
        for sp in self.specs:
            g = _group_of(sp)
            if g not in forms:
                box = QtWidgets.QGroupBox(GROUP_TITLES.get(g, g))
                forms[g] = QtWidgets.QFormLayout(box)
                lay.addWidget(box)
            w = self._make_widget(sp, values.get(sp['name']) or [sp['default']])
            label = QtWidgets.QLabel(sp.get('label') or sp['name'])
            tip = sp.get('help') or ''
            tip = (tip + '  ' if tip else '') + f"default: {fmt_value(sp['default'])}"
            label.setToolTip(tip)
            w.setToolTip(tip)
            forms[g].addRow(label, w)
        if not self.specs:
            lay.addWidget(QtWidgets.QLabel('This detector has no settable parameters.'))
        self.lbl_count = QtWidgets.QLabel('')
        lay.addWidget(self.lbl_count)
        btns = QtWidgets.QDialogButtonBox(BTN_OK | BTN_CANCEL | BTN_DEFAULTS)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        btns.button(BTN_DEFAULTS).clicked.connect(self.reset_defaults)
        lay.addWidget(btns)
        self._update_count()

    def _make_widget(self, sp, current):
        kind = sp['kind']
        if kind in ('choice', 'bool'):
            options = sp['choices'] if kind == 'choice' else [True, False]
            box = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(box)
            grid.setContentsMargins(0, 0, 0, 0)
            box._items = []
            for i, opt in enumerate(options):
                cb = QtWidgets.QCheckBox(fmt_value(opt))
                cb.setChecked(opt in current)
                cb.toggled.connect(self._update_count)
                grid.addWidget(cb, i // 3, i % 3)
                box._items.append((cb, opt))
            self.widgets[sp['name']] = box
            return box
        edit = QtWidgets.QLineEdit(', '.join(fmt_value(v) for v in current))
        edit.setPlaceholderText(fmt_value(sp['default']))
        edit.textChanged.connect(self._update_count)
        self.widgets[sp['name']] = edit
        return edit

    def _parse(self, sp, text):
        toks = [t.strip() for t in text.split(',') if t.strip()]
        if not toks:
            return [sp['default']]
        out = []
        for t in toks:
            try:
                if sp['kind'] == 'int':
                    f = float(t)
                    if f != int(f):
                        raise ValueError
                    out.append(int(f))
                elif sp['kind'] == 'float':
                    out.append(float(t))
                else:
                    out.append(t)
            except ValueError:
                raise ValueError(f"{sp.get('label') or sp['name']}: {t!r} is not a valid "
                                 f"{'whole number' if sp['kind'] == 'int' else 'number'}")
        return list(dict.fromkeys(out))

    def values(self):
        """{param: [values]} for every parameter; raises ValueError if invalid."""
        out = {}
        for sp in self.specs:
            w = self.widgets[sp['name']]
            if sp['kind'] in ('choice', 'bool'):
                vals = [opt for cb, opt in w._items if cb.isChecked()]
                if not vals:
                    raise ValueError(f"{sp.get('label') or sp['name']}: tick at least one value")
            else:
                vals = self._parse(sp, w.text())
            out[sp['name']] = vals
        return out

    def reset_defaults(self):
        for sp in self.specs:
            w = self.widgets[sp['name']]
            if sp['kind'] in ('choice', 'bool'):
                for cb, opt in w._items:
                    cb.setChecked(opt == sp['default'])
            else:
                w.setText(fmt_value(sp['default']))

    def _update_count(self, *_):
        try:
            n, passes = count_variants(self.specs, self.values())
        except ValueError as e:
            self.lbl_count.setText(f'⚠ {e}')
            return
        extra = ''.join(f' · {GROUP_TITLES[k].split(" ")[0]}: {v} pass(es)' for k, v in passes.items())
        self.lbl_count.setText(f'{n} variant(s) of {self.name}{extra}')

    def _accept(self):
        try:
            self.values()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Parameters', str(e))
            return
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# main window
# ─────────────────────────────────────────────────────────────────────────────
class PackDialog(QtWidgets.QDialog):
    """Settings for packing the current job for the GPU server. Folders that
    stay the same (server weights, code path, drive mapping, GPU request) are
    remembered between sessions."""

    DEFAULTS = {'pack/local': '', 'pack/server': '', 'pack/share_local': 'V:\\',
                'pack/share_server': '/maintenance/',
                'pack/weights': '/maintenance/ICIGDev/GPUPOC/input/dqe/imw_runtime/imw_cache',
                'pack/cache': '', 'pack/code': '/maintenance/ICIGDev/GPUPOC/exe/anup/nisar/g1a_automatch/automatch_gpuaas.py',
                'pack/gpu_mb': 40000, 'pack/windows': '', 'pack/max_feat': 0,
                'pack/all_det': False, 'pack/all_match': False}

    def __init__(self, settings, job, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pack for the GPU server')
        self.settings = settings
        self.job = job
        get = (lambda k: settings.value(k, self.DEFAULTS[k]))
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            'Copies the image, only the reference tiles it can need, the ground truth and a settings file with '
            'server paths into a folder the server can read, and writes the curl requests.'))
        f = QtWidgets.QFormLayout()
        self.local = PathRow('dir')
        self.local.setText(get('pack/local'))
        self.local.edit.setPlaceholderText(r'V:\ICIGDev\GPUPOC\input\dqe\g1a\<set>')
        self.local.edit.setToolTip('Where to copy the scene: a folder on a drive the GPU server also sees '
                                   '(V: here is /maintenance/ there). One folder per scene / test set.')
        f.addRow('Copy to (folder on V:)', self.local)
        self.server = QtWidgets.QLineEdit(get('pack/server'))
        self.server.setPlaceholderText('/maintenance/ICIGDev/GPUPOC/input/dqe/g1a/<set>')
        self.server.setToolTip('The same folder as the server sees it; filled in from the share mapping below.')
        f.addRow('Same folder, server path', self.server)
        self.share_local = QtWidgets.QLineEdit(get('pack/share_local'))
        self.share_server = QtWidgets.QLineEdit(get('pack/share_server'))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.share_local)
        row.addWidget(QtWidgets.QLabel('is'))
        row.addWidget(self.share_server)
        f.addRow('Share mapping', row)
        self.weights = QtWidgets.QLineEdit(get('pack/weights'))
        f.addRow('Weights folder on the server', self.weights)
        self.cache = QtWidgets.QLineEdit(get('pack/cache'))
        self.cache.setPlaceholderText('optional: reused preprocessing cache, e.g. /maintenance/.../inter/g1a')
        f.addRow('Cache folder on the server', self.cache)
        self.code = QtWidgets.QLineEdit(get('pack/code'))
        f.addRow('automatch_gpuaas.py on the server', self.code)
        self.gpu_mb = QtWidgets.QSpinBox()
        self.gpu_mb.setRange(500, 200000)
        self.gpu_mb.setSingleStep(1000)
        self.gpu_mb.setSuffix(' MB')
        self.gpu_mb.setValue(int(get('pack/gpu_mb')))
        self.gpu_mb.setToolTip('max_gpu_mem_required: 40000 for the A100, ~19000 for the A4500')
        f.addRow('GPU memory to request', self.gpu_mb)
        self.windows = QtWidgets.QLineEdit(get('pack/windows'))
        self.windows.setPlaceholderText(f"as in the job ({fmt_list(job['window_sizes'])}); e.g. 3072 on the A100")
        f.addRow('Window sizes on the server', self.windows)
        self.max_feat = QtWidgets.QSpinBox()
        self.max_feat.setRange(0, 1000000)
        self.max_feat.setSingleStep(1000)
        self.max_feat.setSpecialValueText(f"as in the job ({job['max_num_features']})")
        self.max_feat.setValue(int(get('pack/max_feat')))
        f.addRow('Max keypoints per window', self.max_feat)
        self.all_det = QtWidgets.QCheckBox('every detector the server offers (kornia + imcui)')
        self.all_det.setChecked(str(get('pack/all_det')).lower() == 'true')
        self.all_match = QtWidgets.QCheckBox('every matcher each detector offers')
        self.all_match.setChecked(str(get('pack/all_match')).lower() == 'true')
        f.addRow('Detectors', self.all_det)
        f.addRow('Matchers', self.all_match)
        self.dry = QtWidgets.QCheckBox('only list what would be copied, and its size')
        f.addRow('', self.dry)
        lay.addLayout(f)
        bb = _DBB(BTN_OK | BTN_CANCEL)
        bb.button(BTN_OK).setText('Pack')
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.local.edit.textChanged.connect(self._suggest_server)

    def _suggest_server(self, text):
        """V:\\X\\Y -> /maintenance/X/Y with the share mapping."""
        lp, sp = self.share_local.text(), self.share_server.text()
        if lp and text.lower().startswith(lp.lower()):
            rest = text[len(lp):].replace('\\', '/').strip('/')
            self.server.setText(sp.rstrip('/') + '/' + rest)

    def args(self):
        a = ['--to', self.local.text(), '--server-dir', self.server.text().strip(),
             '--server-weights', self.weights.text().strip(), '--code-path', self.code.text().strip(),
             '--gpu-mb', str(self.gpu_mb.value())]
        if self.cache.text().strip():
            a += ['--server-cache', self.cache.text().strip()]
        if self.windows.text().strip():
            a += ['--set', f'window_sizes={json.dumps(parse_int_list(self.windows.text(), "window sizes"))}']
        if self.max_feat.value():
            a += ['--set', f'max_num_features={self.max_feat.value()}']
        if self.all_det.isChecked():
            a += ['--set', 'detectors=all']
        if self.all_match.isChecked():
            a += ['--set', 'matchers=all']
        if self.dry.isChecked():
            a.append('--dry-run')
        return a, self.job

    def _accept(self):
        if not self.local.text() or not self.server.text().strip().startswith('/'):
            QtWidgets.QMessageBox.warning(self, 'Pack', 'Give the folder on the share, and the same folder '
                                                        'as the server sees it (starting with /).')
            return
        try:
            self.args()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Pack', str(e))
            return
        for key, val in (('pack/local', self.local.text()), ('pack/server', self.server.text().strip()),
                         ('pack/share_local', self.share_local.text()), ('pack/share_server', self.share_server.text()),
                         ('pack/weights', self.weights.text().strip()), ('pack/cache', self.cache.text().strip()),
                         ('pack/code', self.code.text().strip()), ('pack/gpu_mb', self.gpu_mb.value()),
                         ('pack/windows', self.windows.text().strip()), ('pack/max_feat', self.max_feat.value()),
                         ('pack/all_det', self.all_det.isChecked()), ('pack/all_match', self.all_match.isChecked())):
            self.settings.setValue(key, val)
        self.accept()


class ReportDialog(QtWidgets.QDialog):
    """A read-only text report with a Copy button."""

    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 600)
        lay = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit(text)
        self.text.setReadOnly(True)
        mono = QtGui.QFont('Monospace')
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter if QT_API != 'PyQt5' else QtGui.QFont.TypeWriter)
        self.text.setFont(mono)
        lay.addWidget(self.text, 1)
        row = QtWidgets.QHBoxLayout()
        copy = QtWidgets.QPushButton('Copy all')
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.text.toPlainText()))
        row.addWidget(copy)
        row.addStretch(1)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)


class WeightsDialog(QtWidgets.QDialog):
    """Result of a weights check: one row per detector x weight file."""

    COLS = ['detector', 'status', 'file', 'model', 'found in / download from', 'needed by']

    def __init__(self, ev, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Model weights')
        self.resize(1100, 600)
        self.ev = ev
        lay = QtWidgets.QVBoxLayout(self)
        s = ev.get('summary', {})
        head = QtWidgets.QLabel(
            f"{s.get('files', 0)} weight file(s) for {s.get('detectors', 0)} detector(s): "
            f"{s.get('found', 0)} found, {s.get('missing', 0)} missing or damaged "
            f"({ev.get('mode', 'check')} mode).\nSearched: "
            + '; '.join(ev.get('torch_folders', []) + ev.get('hf_caches', []))
            + (f"\nPut missing torch-hub files, under the name shown, in: {ev.get('put_in')}"
               if ev.get('missing') else ''))
        head.setWordWrap(True)
        head.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse if QT_API != 'PyQt5'
                                     else Qt.TextSelectableByMouse)
        lay.addWidget(head)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setEditTriggers(NO_EDIT)
        self.table.setSelectionBehavior(SEL_ROWS)
        bad_brush = QtGui.QBrush(QtGui.QColor(255, 215, 215))
        for d in ev.get('detectors', []):
            files = d.get('files') or [{'status': d['status'], 'file': '(none needed)' if d['status'] ==
                                        'no weights' else '', 'model': '', 'source': '', 'folder': '',
                                        'needed_by': [], 'all_variants': True}]
            for f in files:
                r = self.table.rowCount()
                self.table.insertRow(r)
                where = f.get('folder') if f['status'] == 'ok' else f.get('source', '')
                need = 'all variants' if f.get('all_variants') else ', '.join(f.get('needed_by', []))
                vals = [d['detector'], f['status'], f.get('file', ''), f.get('model', ''), where, need]
                for c, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(str(v))
                    it.setToolTip(str(v))
                    if f['status'] not in ('ok', 'no weights'):
                        it.setBackground(bad_brush)
                    self.table.setItem(r, c, it)
            for v in d.get('variants', []):
                if v.get('status') in ('error', 'partial'):
                    r = self.table.rowCount()
                    self.table.insertRow(r)
                    for c, val in enumerate([d['detector'], v['status'], '', '', v.get('error', ''),
                                             v.get('variant', '')]):
                        it = QtWidgets.QTableWidgetItem(str(val))
                        it.setToolTip(str(val))
                        it.setBackground(bad_brush)
                        self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        copy = QtWidgets.QPushButton('Copy missing list')
        copy.setEnabled(bool(ev.get('missing')))
        copy.clicked.connect(self.copy_missing)
        row.addWidget(copy)
        row.addStretch(1)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def missing_text(self) -> str:
        return '\n'.join(f"{m['file']}\t{m['source']}\t{', '.join(m.get('detectors', []))}"
                         for m in self.ev.get('missing', []))

    def copy_missing(self):
        QtWidgets.QApplication.clipboard().setText(self.missing_text())


class TruthDialog(QtWidgets.QDialog):
    """Detector + matcher ranking against the ground truth (best setting each)."""

    COLS = [('rank', '#'), ('channel', 'channel'), ('detector', 'detector'), ('matcher_family', 'matcher'),
            ('best_matcher_setting', 'setting'), ('ransac', 'RANSAC'), ('sweep', 'sweep'),
            ('truth_rmse_m', 'RMSE m'), ('truth_mean_dE_m', 'bias dE m'), ('truth_mean_dN_m', 'bias dN m'),
            ('truth_max_m', 'max m'), ('truth_reached', 'reached'), ('truth_total', 'of'),
            ('n_chips', 'chips')]

    def __init__(self, ev, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Detector + matcher vs ground truth')
        self.resize(1100, 520)
        lay = QtWidgets.QVBoxLayout(self)
        files = ev.get('files') or {}
        head = QtWidgets.QLabel(
            f"{ev.get('n_truth', '?')} ground-truth point(s). Each detector + matcher at its best setting; "
            f"ranked by truth points reached, then RMSE of (tool - truth).\nAll configurations: "
            f"{files.get('TRUTH_RANKING', '')}\nPer truth point: {files.get('TRUTH_POINTS', '')}")
        head.setWordWrap(True)
        lay.addWidget(head)
        t = QtWidgets.QTableWidget(0, len(self.COLS))
        t.setHorizontalHeaderLabels([c[1] for c in self.COLS])
        t.setEditTriggers(NO_EDIT)
        t.setSelectionBehavior(SEL_ROWS)
        for row in ev.get('top') or []:
            r = t.rowCount()
            t.insertRow(r)
            for c, (key, _) in enumerate(self.COLS):
                v = row.get(key)
                txt = '' if v is None else (f'{v:.1f}' if isinstance(v, float) else str(v))
                t.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        t.resizeColumnsToContents()
        self.table = t
        lay.addWidget(t, 1)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.accept)
        lay.addWidget(close)


class AutoMatchWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DPQED AutoMatch — geolocation vs reference')
        self.resize(1250, 900)
        self.proc = None
        self.proc_kind = None
        self.proc_buffer = ''
        self._queue = []              # quick commands waiting for the process
        self._pack_lines, self._pack_dry, self._pack_dest = [], False, ''
        self.detector_info = []       # [{'name','source','matchers','params'}]
        self.detector_specs = {}      # name -> [param spec dicts]
        self.detector_defaults = {}   # name -> matchers ticked by default
        self.detector_params = {}     # name -> {param: [values]} (non-default only)
        self.settings = QtCore.QSettings('DPQED', 'AutoMatch')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical if QT_API != 'PyQt5' else Qt.Vertical)
        root.addWidget(split, 1)

        self.tabs = QtWidgets.QTabWidget()
        split.addWidget(self.tabs)
        self._build_data_tab()
        self._build_algo_tab()
        self._build_offset_tab()
        self._build_ransac_tab()
        self._build_run_tab()

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        btns = QtWidgets.QHBoxLayout()
        for label, slot in (('Load job…', self.load_job_dialog), ('Save job…', self.save_job_dialog),
                            ('Preflight', self.preflight), ('Run', self.run_job),
                            ('Stop', self.stop), ('Open output', self.open_output),
                            ('Pack for server…', self.pack_dialog)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            btns.addWidget(b)
            setattr(self, 'btn_' + label.split()[0].lower().rstrip('…'), b)
        self.btn_stop.setEnabled(False)
        bl.addLayout(btns)
        self.lbl_status = QtWidgets.QLabel('Ready.')
        bl.addWidget(self.lbl_status)
        self.bar_sweep = QtWidgets.QProgressBar()
        self.bar_sweep.setFormat('sweep %v / %m')
        self.bar_step = QtWidgets.QProgressBar()
        self.bar_step.setFormat('%p%')
        bl.addWidget(self.bar_sweep)
        bl.addWidget(self.bar_step)

        self.results = QtWidgets.QTableWidget(0, len(RESULT_COLS))
        self.results.setHorizontalHeaderLabels(RESULT_COLS)
        self.results.setSelectionBehavior(SEL_ROWS)
        self.results.setEditTriggers(NO_EDIT)
        self.results.cellDoubleClicked.connect(self._open_result)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        mono = QtGui.QFont('Monospace')
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter if QT_API != 'PyQt5' else QtGui.QFont.TypeWriter)
        self.log.setFont(mono)
        out_tabs = QtWidgets.QTabWidget()
        out_tabs.addTab(self.results, 'Results')
        out_tabs.addTab(self.log, 'Log')
        bl.addWidget(out_tabs, 1)
        split.addWidget(bottom)
        split.setSizes([480, 420])

        self.set_job(DEFAULT_JOB)
        last = self.settings.value('last_job')
        if last:
            try:
                self.set_job({**DEFAULT_JOB, **json.loads(last)})
            except Exception:
                pass
        QtCore.QTimer.singleShot(0, self.refresh_detectors)

    # ── tabs ─────────────────────────────────────────────────────────────────
    def _form_tab(self, title):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.tabs.addTab(w, title)
        return form

    def _build_data_tab(self):
        f = self._form_tab('Data')
        self.in_path = PathRow('file_or_dir', 'Rasters/HDF5 (*.tif *.tiff *.vrt *.jp2 *.img *.h5 *.hdf5);;All (*)')
        f.addRow('Input image (G1A / NISAR)', self.in_path)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Inspect input')
        b.clicked.connect(self.inspect_input)
        row.addWidget(b)
        self.lbl_input = QtWidgets.QLabel('—')
        self.lbl_input.setWordWrap(True)
        row.addWidget(self.lbl_input, 1)
        f.addRow(row)
        self.channels = QtWidgets.QListWidget()
        self.channels.setMaximumHeight(90)
        f.addRow('Channels (none checked = all)', self.channels)

        self.ref_path = PathRow('dir')
        f.addRow('Reference folder (L8_ref / C1 / …)', self.ref_path)
        self.ref_mode = QtWidgets.QComboBox()
        self.ref_mode.addItems(REF_MODES)
        self.ref_label = QtWidgets.QLineEdit()
        self.ref_label.setPlaceholderText('default: folder name')
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel('mode'))
        hl.addWidget(self.ref_mode)
        hl.addWidget(QtWidgets.QLabel('label'))
        hl.addWidget(self.ref_label)
        f.addRow('Reference discovery', hl)
        self.ref_band = QtWidgets.QSpinBox()
        self.ref_band.setRange(1, 64)
        self.ref_band_map = QtWidgets.QLineEdit()
        self.ref_band_map.setPlaceholderText('optional per channel, e.g. band1:4, band2:3')
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.ref_band)
        hl.addWidget(self.ref_band_map, 1)
        f.addRow('Reference band', hl)
        self.ref_fill = QtWidgets.QLineEdit()
        self.ref_scale = QtWidgets.QDoubleSpinBox()
        self.ref_scale.setDecimals(6)
        self.ref_scale.setRange(1e-6, 1e6)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel('fill values'))
        hl.addWidget(self.ref_fill)
        hl.addWidget(QtWidgets.QLabel('DN scale'))
        hl.addWidget(self.ref_scale)
        f.addRow('Reference values', hl)

        self.out_path = PathRow('dir')
        f.addRow('Output folder', self.out_path)
        self.tmp_path = PathRow('dir')
        self.tmp_path.edit.setPlaceholderText('default: <output>/_cache')
        f.addRow('Cache folder', self.tmp_path)
        self.gcp_path = PathRow('file', 'CSV (*.csv);;All (*)')
        self.gcp_path.edit.setToolTip('NISAR-style GCPs (columns scan, pix, Map_X, Map_Y, Map_X_ref, Map_Y_ref) '
                                      'that steer the chip consensus. Manually measured RIVAL points go in '
                                      'Ground truth CSV below; leave this empty when choosing a detector.')
        f.addRow('Manual GCP CSV (optional)', self.gcp_path)
        self.truth_path = PathRow('file', 'CSV (*.csv);;All (*)')
        self.truth_path.edit.setToolTip('Manually measured points as a RIVAL CSV (In/Ref lon/lat columns). '
                                        'After the run every detector + matcher is ranked against them '
                                        '(TRUTH_BY_DETECTOR_MATCHER.csv). Evaluation only: it does not '
                                        'steer the matching.')
        f.addRow('Ground truth CSV (optional)', self.truth_path)
        self.truth_radius = QtWidgets.QDoubleSpinBox()
        self.truth_radius.setRange(0.5, 100.0)
        self.truth_radius.setSuffix(' km')
        self.truth_radius.setToolTip('Tool matches this close to a truth point estimate its error there '
                                     '(else the tool\'s error surface is used)')
        f.addRow('Truth neighbourhood', self.truth_radius)

    def _build_algo_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('kornia first; imcui adds what kornia lacks (greyed rows: not '
                                       'offered, with the reason). Unticked matchers run only if ticked.'))
        top.addStretch(1)
        b = QtWidgets.QPushButton('Refresh list')
        b.clicked.connect(self.refresh_detectors)
        top.addWidget(b)
        wb = QtWidgets.QToolButton()
        wb.setText('Check weights')
        wb.setToolTip('Which model weight files the detectors need, and whether they are on this '
                      'machine (nothing is downloaded)')
        menu = QtWidgets.QMenu(wb)
        menu.addAction('Selected detectors (as configured)', lambda: self.check_weights('selected'))
        menu.addAction('All detectors, every weight choice', lambda: self.check_weights('all'))
        menu.addAction('Selected detectors — load each model (slower)',
                       lambda: self.check_weights('selected', load=True))
        wb.setMenu(menu)
        wb.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup if QT_API != 'PyQt5'
                        else QtWidgets.QToolButton.InstantPopup)
        top.addWidget(wb)
        self.btn_weights = wb
        lay.addLayout(top)
        self.det_table = QtWidgets.QTableWidget(0, 5)
        self.det_table.setHorizontalHeaderLabels(['detector', 'source', 'matchers (check to run)',
                                                  'parameters', 'weights'])
        self.det_table.horizontalHeader().setStretchLastSection(True)
        self.det_table.setEditTriggers(NO_EDIT)
        lay.addWidget(self.det_table, 1)
        form = QtWidgets.QFormLayout()
        self.smnn = QtWidgets.QLineEdit()
        form.addRow('SMNN thresholds', self.smnn)
        self.weights_cache = PathRow('dir')
        self.weights_cache.edit.setToolTip('Extra folder searched for model weights (kornia and imcui), e.g. '
                                           'the output of prefetch_weights.py. kornia\'s own default '
                                           'folder is always searched too.')
        form.addRow('Weights folder (offline)', self.weights_cache)
        self.lbl_imcui = QtWidgets.QLabel('')
        self.lbl_imcui.setWordWrap(True)
        form.addRow(self.lbl_imcui)
        lay.addLayout(form)
        self.tabs.addTab(w, 'Detectors & matchers')

    def _build_offset_tab(self):
        f = self._form_tab('Windows & offsets')
        self.windows = QtWidgets.QLineEdit()
        f.addRow('Window sizes (px)', self.windows)
        self.nfeat = QtWidgets.QLineEdit()
        f.addRow('Keypoints per window (list, "auto")', self.nfeat)
        self.kp_density = QtWidgets.QSpinBox()
        self.kp_density.setRange(100, 200000)
        f.addRow('Keypoint density for auto (/Mpx)', self.kp_density)
        self.max_feat = QtWidgets.QSpinBox()
        self.max_feat.setRange(1000, 1000000)
        self.max_feat.setSingleStep(1000)
        self.max_feat.setToolTip('Cap on keypoints per window. Raise it to keep the density at large windows '
                                 '(e.g. 90000 for 3072 px on a 40 GB GPU); matching is done in slices, so '
                                 'memory stays bounded, but time grows with the count.')
        f.addRow('Max keypoints per window', self.max_feat)
        self.lbl_kp = QtWidgets.QLabel('')
        self.lbl_kp.setWordWrap(True)
        f.addRow('', self.lbl_kp)
        for w in (self.windows, self.nfeat):
            w.textChanged.connect(self._update_kp_label)
        for w in (self.kp_density, self.max_feat):
            w.valueChanged.connect(self._update_kp_label)
        self.target_res = QtWidgets.QLineEdit()
        self.target_res.setPlaceholderText('native')
        f.addRow('Working resolution (m, blank = native)', self.target_res)
        self.max_err = QtWidgets.QDoubleSpinBox()
        self.max_err.setRange(0, 500)
        self.max_err.setSuffix(' km')
        f.addRow('Max expected error', self.max_err)
        self.coarse = QtWidgets.QComboBox()
        self.coarse.addItems(COARSE_CHOICES)
        f.addRow('Coarse alignment', self.coarse)
        self.coarse_res = QtWidgets.QDoubleSpinBox()
        self.coarse_res.setRange(5, 5000)
        self.coarse_res.setSuffix(' m')
        f.addRow('Coarse resolution', self.coarse_res)
        self.use_init = QtWidgets.QCheckBox('use')
        self.init_dx = QtWidgets.QDoubleSpinBox()
        self.init_dy = QtWidgets.QDoubleSpinBox()
        for s in (self.init_dx, self.init_dy):
            s.setRange(-1e6, 1e6)
            s.setDecimals(1)
            s.setSuffix(' m')
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.use_init)
        hl.addWidget(QtWidgets.QLabel('dE'))
        hl.addWidget(self.init_dx)
        hl.addWidget(QtWidgets.QLabel('dN'))
        hl.addWidget(self.init_dy)
        f.addRow('Known offset (input − reference)', hl)
        self.margin = QtWidgets.QDoubleSpinBox()
        self.margin.setRange(0, 50000)
        self.margin.setSuffix(' m')
        f.addRow('Fine search margin', self.margin)
        self.coarse_local = QtWidgets.QCheckBox('estimate the offset per cell (errors that vary across the scene)')
        f.addRow('Local coarse offsets', self.coarse_local)
        self.coarse_cell = QtWidgets.QDoubleSpinBox()
        self.coarse_cell.setRange(0, 500)
        self.coarse_cell.setDecimals(1)
        self.coarse_cell.setSuffix(' km')
        self.coarse_cell.setSpecialValueText('one window')
        f.addRow('Offset cell size', self.coarse_cell)

    def _build_ransac_tab(self):
        f = self._form_tab('RANSAC & consensus')
        hl = QtWidgets.QHBoxLayout()
        self.ransac_boxes = {}
        for m in RANSAC_CHOICES:
            cb = QtWidgets.QCheckBox(m)
            self.ransac_boxes[m] = cb
            hl.addWidget(cb)
        f.addRow('Estimators', hl)
        self.ransac_px = QtWidgets.QLineEdit()
        f.addRow('Thresholds (pixels)', self.ransac_px)
        self.ransac_conf = QtWidgets.QLineEdit()
        f.addRow('Confidences', self.ransac_conf)
        self.cons_tol = QtWidgets.QDoubleSpinBox()
        self.cons_tol.setRange(0.1, 1000)
        self.cons_tol.setSuffix(' px')
        f.addRow('Consensus tolerance', self.cons_tol)
        self.cons_bin = QtWidgets.QDoubleSpinBox()
        self.cons_bin.setRange(0.05, 100)
        self.cons_bin.setSuffix(' px')
        f.addRow('Consensus mode bin', self.cons_bin)
        self.min_inl = QtWidgets.QSpinBox()
        self.min_inl.setRange(3, 100000)
        f.addRow('Min inliers per chip', self.min_inl)
        self.min_chips = QtWidgets.QSpinBox()
        self.min_chips.setRange(1, 10000)
        f.addRow('Min surviving chips', self.min_chips)
        self.cons_model = QtWidgets.QComboBox()
        self.cons_model.addItems(CONSENSUS_MODELS)
        self.cons_model.setToolTip('surface: chips must agree with a smooth error surface (error varies '
                                   'across the scene); constant: with the most common error; none: keep all')
        f.addRow('Chip consistency', self.cons_model)
        self.cons_surface = QtWidgets.QComboBox()
        self.cons_surface.addItems(SURFACE_DEGREES)
        f.addRow('Error surface', self.cons_surface)

    def _build_run_tab(self):
        f = self._form_tab('Output & run')
        self.rival_pts = QtWidgets.QSpinBox()
        self.rival_pts.setRange(0, 1000000)
        self.rival_pts.setSpecialValueText('all')
        f.addRow('RIVAL points per chip', self.rival_pts)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(['auto', 'cuda', 'cuda:0', 'cuda:1', 'cpu'])
        self.device.setEditable(True)
        f.addRow('Device', self.device)
        self.min_gpu = QtWidgets.QDoubleSpinBox()
        self.min_gpu.setRange(0, 80)
        self.min_gpu.setSuffix(' GB')
        f.addRow('Min free GPU memory per window', self.min_gpu)
        self.min_area = QtWidgets.QDoubleSpinBox()
        self.min_area.setRange(0, 1e6)
        self.min_area.setSuffix(' km²')
        f.addRow('Min overlap area', self.min_area)
        self.min_valid = QtWidgets.QDoubleSpinBox()
        self.min_valid.setRange(0.01, 1.0)
        self.min_valid.setSingleStep(0.05)
        f.addRow('Min valid fraction per window', self.min_valid)
        self.amp = QtWidgets.QCheckBox('mixed precision (AMP)')
        self.resume = QtWidgets.QCheckBox('resume finished sweep points')
        self.save_imgs = QtWidgets.QCheckBox('save match preview images (imcui)')
        self.debug = QtWidgets.QCheckBox('debug logging')
        for cb in (self.amp, self.resume, self.save_imgs, self.debug):
            f.addRow(cb)

    # ── job <-> widgets ──────────────────────────────────────────────────────
    def set_job(self, job):
        j = {**DEFAULT_JOB, **job}
        self.in_path.setText(j['input_path'])
        self.ref_path.setText(j['reference_dir'])
        self.ref_mode.setCurrentText(j['reference_mode'] or 'auto')
        self.ref_label.setText(j['reference_label'])
        self.ref_band.setValue(int(j['reference_band']))
        self.ref_band_map.setText(', '.join(f'{k}:{v}' for k, v in (j['reference_band_map'] or {}).items()))
        self.ref_fill.setText(fmt_list(j['reference_fill_values']))
        self.ref_scale.setValue(float(j['reference_scale']))
        self.out_path.setText(j['output_dir'])
        self.tmp_path.setText(j['temp_dir'])
        self.gcp_path.setText(j['manual_gcp_csv'])
        self.truth_path.setText(j.get('truth_csv') or '')
        self.truth_radius.setValue(float(j.get('truth_radius_m') or 5000) / 1000.0)
        self._pending_channels = list(j['channels'] or [])
        self._set_channels([self.channels.item(i).text() for i in range(self.channels.count())]
                           or self._pending_channels)
        self.detector_params = {k: {p: list(v) for p, v in d.items()}
                                for k, d in (j.get('detector_params') or {}).items()}
        self._pending_detectors = (list(j['detectors']), dict(j['matchers'] or {}))
        self._apply_detector_selection()
        self._refresh_param_summaries()
        self.smnn.setText(fmt_list(j['smnn_thresholds']))
        self.weights_cache.setText(j['weights_cache_dir'])
        self.windows.setText(fmt_list(j['window_sizes']))
        self.nfeat.setText(fmt_list(j['num_features']))
        self.kp_density.setValue(int(j['keypoint_density']))
        self.max_feat.setValue(int(j.get('max_num_features') or 32000))
        self._update_kp_label()
        self.target_res.setText('' if j['target_resolution'] is None else f"{j['target_resolution']:g}")
        self.max_err.setValue(float(j['max_expected_error_m']) / 1000.0)
        self.coarse.setCurrentText(j['coarse_method'])
        self.coarse_res.setValue(float(j['coarse_resolution_m']))
        off = j['initial_offset_m']
        self.use_init.setChecked(bool(off))
        self.init_dx.setValue(float(off[0]) if off else 0.0)
        self.init_dy.setValue(float(off[1]) if off else 0.0)
        self.margin.setValue(float(j['search_margin_m']))
        self.coarse_local.setChecked(bool(j['coarse_local']))
        self.coarse_cell.setValue(float(j['coarse_cell_km'] or 0))
        for m, cb in self.ransac_boxes.items():
            cb.setChecked(m in [str(x) for x in j['ransac_methods']])
        self.ransac_px.setText(fmt_list(j['ransac_thresholds_px']))
        self.ransac_conf.setText(fmt_list(j['ransac_confidences']))
        self.cons_tol.setValue(float(j['consensus_tolerance_px']))
        self.cons_bin.setValue(float(j['consensus_mode_bin_px']))
        self.min_inl.setValue(int(j['min_inliers_per_chip']))
        self.min_chips.setValue(int(j['min_surviving_chips']))
        self.cons_model.setCurrentText(j['consensus_model'])
        self.cons_surface.setCurrentText(j['consensus_surface'])
        self.rival_pts.setValue(int(j['rival_max_points_per_chip']))
        self.device.setCurrentText(j['device'])
        self.min_gpu.setValue(float(j['min_gpu_free_gb']))
        self.min_area.setValue(float(j['min_area_km2']))
        self.min_valid.setValue(float(j['min_valid_fraction']))
        self.amp.setChecked(bool(j['use_amp']))
        self.resume.setChecked(bool(j['resume']))
        self.save_imgs.setChecked(bool(j['save_match_images']))
        self.debug.setChecked(bool(j['debug']))

    def get_job(self):
        """Widgets -> job dict. Raises ValueError with a readable message."""
        dets, matchers = self._selected_detectors()
        if not dets:
            raise ValueError('Select at least one detector (Detectors & matchers tab).')
        methods = [m for m, cb in self.ransac_boxes.items() if cb.isChecked()]
        if not methods:
            raise ValueError('Select at least one RANSAC estimator.')
        tr = self.target_res.text().strip()
        job = dict(DEFAULT_JOB)
        job.update({
            'input_path': self.in_path.text(),
            'channels': [self.channels.item(i).text() for i in range(self.channels.count())
                         if self.channels.item(i).checkState() == CHECKED],
            'reference_dir': self.ref_path.text(),
            'reference_label': self.ref_label.text().strip(),
            'reference_mode': self.ref_mode.currentText(),
            'reference_band': self.ref_band.value(),
            'reference_band_map': parse_band_map(self.ref_band_map.text()),
            'reference_fill_values': parse_float_list(self.ref_fill.text(), 'fill values'),
            'reference_scale': self.ref_scale.value(),
            'output_dir': self.out_path.text(),
            'temp_dir': self.tmp_path.text(),
            'manual_gcp_csv': self.gcp_path.text(),
            'truth_csv': self.truth_path.text().strip(),
            'truth_radius_m': self.truth_radius.value() * 1000.0,
            'detectors': dets,
            'matchers': matchers,
            'detector_params': {k: v for k, v in self.detector_params.items() if v},
            'smnn_thresholds': parse_float_list(self.smnn.text(), 'SMNN thresholds'),
            'weights_cache_dir': self.weights_cache.text(),
            'window_sizes': parse_int_list(self.windows.text(), 'window sizes'),
            'num_features': parse_int_list(self.nfeat.text(), 'keypoints', allow_auto=True) or [None],
            'keypoint_density': self.kp_density.value(),
            'max_num_features': self.max_feat.value(),
            'target_resolution': float(tr) if tr else None,
            'max_expected_error_m': self.max_err.value() * 1000.0,
            'coarse_method': self.coarse.currentText(),
            'coarse_resolution_m': self.coarse_res.value(),
            'initial_offset_m': [self.init_dx.value(), self.init_dy.value()] if self.use_init.isChecked() else None,
            'search_margin_m': self.margin.value(),
            'coarse_local': self.coarse_local.isChecked(),
            'coarse_cell_km': self.coarse_cell.value(),
            'ransac_methods': methods,
            'ransac_thresholds_px': parse_float_list(self.ransac_px.text(), 'RANSAC thresholds'),
            'ransac_thresholds_m': [],
            'ransac_confidences': parse_float_list(self.ransac_conf.text(), 'RANSAC confidences'),
            'consensus_tolerance_px': self.cons_tol.value(),
            'consensus_mode_bin_px': self.cons_bin.value(),
            'min_inliers_per_chip': self.min_inl.value(),
            'min_surviving_chips': self.min_chips.value(),
            'consensus_model': self.cons_model.currentText(),
            'consensus_surface': self.cons_surface.currentText(),
            'rival_max_points_per_chip': self.rival_pts.value(),
            'device': self.device.currentText().strip() or 'auto',
            'min_gpu_free_gb': self.min_gpu.value(),
            'min_area_km2': self.min_area.value(),
            'min_valid_fraction': self.min_valid.value(),
            'use_amp': self.amp.isChecked(),
            'resume': self.resume.isChecked(),
            'save_match_images': self.save_imgs.isChecked(),
            'debug': self.debug.isChecked(),
        })
        if not job['window_sizes']:
            raise ValueError('Give at least one window size.')
        return job

    # ── channels ─────────────────────────────────────────────────────────────
    def _set_channels(self, names):
        checked = set(self._pending_channels or [])
        self.channels.clear()
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(USER_CHECKABLE | ITEM_ENABLED)
            it.setCheckState(CHECKED if n in checked else UNCHECKED)
            self.channels.addItem(it)

    # ── detectors ────────────────────────────────────────────────────────────
    def _selectable(self, r):
        """False for the greyed catalogue rows that are not offered."""
        item = self.det_table.item(r, 0)
        return item is not None and bool(item.flags() & USER_CHECKABLE)

    def _selected_detectors(self):
        dets, matchers = [], {}
        for r in range(self.det_table.rowCount()):
            if not self._selectable(r):
                continue
            name = self.det_table.item(r, 0).text()
            if self.det_table.item(r, 0).checkState() != CHECKED:
                continue
            dets.append(name)
            box = self.det_table.cellWidget(r, 2)
            all_m = [cb.text() for cb in box.findChildren(QtWidgets.QCheckBox)]
            on = [cb.text() for cb in box.findChildren(QtWidgets.QCheckBox) if cb.isChecked()]
            if not on:
                raise ValueError(f'{name}: check at least one matcher.')
            if set(on) != set(self.detector_defaults.get(name, all_m)):
                matchers[name] = on
        return dets, matchers

    def _fill_detector_table(self, info, not_offered=()):
        self.detector_info = info
        self.det_table.setRowCount(0)
        for d in info:
            r = self.det_table.rowCount()
            self.det_table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(d['name'])
            it.setFlags(USER_CHECKABLE | ITEM_ENABLED)
            it.setCheckState(UNCHECKED)
            self.det_table.setItem(r, 0, it)
            self.det_table.setItem(r, 1, QtWidgets.QTableWidgetItem(d['source']))
            box = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(box)
            hl.setContentsMargins(4, 0, 4, 0)
            defaults = d.get('default_matchers', d['matchers'])
            self.detector_defaults[d['name']] = list(defaults)
            for m in d['matchers']:
                cb = QtWidgets.QCheckBox(m)
                cb.setChecked(m in defaults)
                if m not in defaults:
                    cb.setToolTip('optional kornia matcher: runs only when ticked')
                hl.addWidget(cb)
            hl.addStretch(1)
            self.det_table.setCellWidget(r, 2, box)
            self.detector_specs[d['name']] = d.get('params') or []
            pbox = QtWidgets.QWidget()
            pl = QtWidgets.QHBoxLayout(pbox)
            pl.setContentsMargins(4, 0, 4, 0)
            btn = QtWidgets.QPushButton('Configure…')
            btn.setEnabled(bool(self.detector_specs[d['name']]))
            btn.clicked.connect(lambda _=False, n=d['name']: self.open_param_dialog(n))
            pl.addWidget(btn)
            summary = QtWidgets.QLabel('')
            summary.setObjectName('summary')
            pl.addWidget(summary, 1)
            self.det_table.setCellWidget(r, 3, pbox)
            self.det_table.setItem(r, 4, QtWidgets.QTableWidgetItem(''))
        # the rest of the catalogue: shown, greyed, with the reason
        for row in not_offered:
            r = self.det_table.rowCount()
            self.det_table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(f"imw-{row['tag']}")
            it.setFlags(Qt.ItemFlag.NoItemFlags if QT_API != 'PyQt5' else Qt.NoItemFlags)
            it.setToolTip(row['reason'])
            self.det_table.setItem(r, 0, it)
            src = QtWidgets.QTableWidgetItem('imcui — not offered')
            src.setFlags(Qt.ItemFlag.NoItemFlags if QT_API != 'PyQt5' else Qt.NoItemFlags)
            self.det_table.setItem(r, 1, src)
            use = QtWidgets.QLabel(f"use kornia {row['use']}" if row.get('use') else '—')
            use.setEnabled(False)
            self.det_table.setCellWidget(r, 2, use)
            why = QtWidgets.QLabel(row['reason'])
            why.setEnabled(False)
            why.setToolTip(row['reason'])
            self.det_table.setCellWidget(r, 3, why)
        self.det_table.resizeColumnsToContents()
        self._apply_detector_selection()
        self._refresh_param_summaries()

    def _apply_detector_selection(self):
        dets, matchers = getattr(self, '_pending_detectors', ([], {}))
        # older job files name the DISK weights as detectors
        legacy = {'disk_depth': 'depth', 'disk_epipolar': 'epipolar'}
        want = set()
        for d in dets:
            if d in legacy:
                want.add('disk')
                ck = self.detector_params.setdefault('disk', {}).setdefault('checkpoint', [])
                if legacy[d] not in ck:
                    ck.append(legacy[d])
            else:
                want.add(d)
        for r in range(self.det_table.rowCount()):
            if not self._selectable(r):
                continue
            name = self.det_table.item(r, 0).text()
            self.det_table.item(r, 0).setCheckState(CHECKED if name in want else UNCHECKED)
            if name in matchers:
                for cb in self.det_table.cellWidget(r, 2).findChildren(QtWidgets.QCheckBox):
                    cb.setChecked(cb.text() in matchers[name])

    # ── detector parameters ──────────────────────────────────────────────────
    def _row_of(self, name):
        for r in range(self.det_table.rowCount()):
            if self.det_table.item(r, 0).text() == name:
                return r
        return None

    def _param_summary(self, name):
        specs = self.detector_specs.get(name) or []
        if not specs:
            return ''
        chosen = prune_params(specs, self.detector_params.get(name) or {})
        if not chosen:
            return 'defaults'
        full = {sp['name']: chosen.get(sp['name'], [sp['default']]) for sp in specs}
        n, passes = count_variants(specs, full)
        labels = {sp['name']: (sp.get('label') or sp['name']) for sp in specs}
        text = '; '.join(f"{labels[k]}: {', '.join(fmt_value(v) for v in vals)}"
                         for k, vals in chosen.items())
        extra = f' → {n} variants' if n > 1 else ''
        extra += ''.join(f', {v} {k} passes' for k, v in passes.items() if v > 1)
        return text + extra

    def _refresh_param_summaries(self):
        for name in self.detector_specs:
            r = self._row_of(name)
            if r is None:
                continue
            lbl = self.det_table.cellWidget(r, 3).findChild(QtWidgets.QLabel, 'summary')
            lbl.setText(self._param_summary(name))
        self.det_table.resizeColumnToContents(3)

    def open_param_dialog(self, name, run=True):
        """Open the parameter dialog of one detector. run=False returns the
        dialog without showing it (tests drive it directly)."""
        specs = self.detector_specs.get(name) or []
        current = {sp['name']: (self.detector_params.get(name) or {}).get(sp['name'], [sp['default']])
                   for sp in specs}
        dlg = ParamDialog(name, specs, current, self)
        dlg.accepted.connect(lambda: self._store_params(name, dlg))
        if run:
            dlg.exec() if hasattr(dlg, 'exec') else dlg.exec_()
        return dlg

    def _store_params(self, name, dlg):
        chosen = prune_params(dlg.specs, dlg.values())
        if chosen:
            self.detector_params[name] = chosen
        else:
            self.detector_params.pop(name, None)
        self._refresh_param_summaries()

    def refresh_detectors(self):
        try:
            self._pending_detectors = self._selected_detectors() if self.det_table.rowCount() else \
                self._pending_detectors
        except ValueError:
            pass
        args = ['detectors']
        if self.weights_cache.text():
            args += ['--weights-cache', self.weights_cache.text()]
        self._start(args, 'detectors')

    # ── subprocess plumbing ──────────────────────────────────────────────────
    def _start(self, args, kind):
        if self.proc is not None and self.proc.state() == PROC_NOT_RUNNING:
            self.proc = None
        if self.proc is not None:
            if kind in ('detectors', 'inspect', 'weights'):
                self._queue = [q for q in self._queue if q[1] != kind] + [(args, kind)]
                self.lbl_status.setText(f'{kind}: queued until {self.proc_kind} finishes')
                return True
            QtWidgets.QMessageBox.information(self, 'Busy', f'Busy ({self.proc_kind}); try again when it finishes.')
            return False
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(MERGED)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        env.insert('PYTHONIOENCODING', 'utf-8')
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self.proc_kind = kind
        self.proc_buffer = ''
        self._events = []
        self.proc.start(sys.executable, [JOB_SCRIPT] + args)
        busy = kind in ('run', 'preflight', 'pack')
        self.btn_run.setEnabled(not busy)
        self.btn_preflight.setEnabled(not busy)
        self.btn_stop.setEnabled(kind in ('run', 'pack'))
        self.lbl_status.setText(f'{kind} …')
        return True

    def _read_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode('utf-8', 'replace')
        self.proc_buffer += data.replace('\r', '\n')
        lines = self.proc_buffer.split('\n')
        self.proc_buffer = lines.pop()
        for line in lines:
            if line.startswith(PROGRESS_PREFIX):
                try:
                    ev = json.loads(line[len(PROGRESS_PREFIX):])
                except ValueError:
                    continue
                self._events.append(ev)
                self._on_event(ev)
            elif line.strip():
                self.log.appendPlainText(line)
                if self.proc_kind == 'pack':
                    self._pack_lines.append(line)
                    m = re.match(r'\[pack\] \((\d+)/(\d+)\) (.*)', line)
                    if m:
                        self.bar_step.setMaximum(int(m.group(2)))
                        self.bar_step.setValue(int(m.group(1)) - (0 if ' done' in line or 'already' in line else 1))
                        self.bar_step.setFormat('copying  %v/%m')
                    if line.startswith('[pack]'):
                        self.lbl_status.setText(line[:160])

    def _finished(self, code, _status=None):
        if self.proc_buffer.strip():
            self.log.appendPlainText(self.proc_buffer)
        kind, events = self.proc_kind, list(self._events)
        self.proc = None
        if self._queue:
            QtCore.QTimer.singleShot(0, lambda: self._start(*self._queue.pop(0)))
        self.btn_run.setEnabled(True)
        self.btn_preflight.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if kind == 'detectors':
            ev = next((e for e in events if e.get('event') == 'detectors'), None)
            if ev:
                imw = ev.get('imcui') or {}
                not_offered = imw.get('not_offered') or [
                    {'tag': t, 'reason': why, 'use': ''} for t, why in imw.get('skipped', [])]
                self._fill_detector_table(ev['detectors'], not_offered)
                txt = imw.get('error') or (f"imcui: {len(imw.get('registered', []))} detector(s) offered")
                n_k = sum(1 for r in not_offered if r.get('kind', 'kornia') == 'kornia')
                n_u = len(not_offered) - n_k
                if n_k:
                    txt += f'; {n_k} not offered because kornia provides them'
                if n_u:
                    txt += f'; {n_u} not in the installed imcui'
                self.lbl_imcui.setText(txt)
                self.lbl_status.setText('Detector list updated.')
            else:
                self.lbl_status.setText('Could not list detectors — see Log.')
        elif kind == 'inspect':
            ev = next((e for e in events if e.get('event') == 'inspect'), None)
            if ev:
                self._set_channels(ev['channels'])
                res = ev.get('native_res')
                self.lbl_input.setText(
                    f"{ev['kind']} · {ev['name']} · {len(ev['channels'])} channel(s) · "
                    f"working CRS {ev['working_crs']}" + (f' · {res:g} m' if res else ''))
                self.lbl_status.setText('Input inspected.')
            else:
                self.lbl_status.setText('Input could not be read — see Log.')
        elif kind == 'preflight':
            ev = next((e for e in events if e.get('event') == 'preflight'), None)
            self._show_preflight(ev)
        elif kind == 'pack':
            self._pack_finished(code)
        elif kind == 'weights':
            tmp_job = getattr(self, '_weights_job_file', None)
            if tmp_job:
                try:
                    os.remove(tmp_job)
                except OSError:
                    pass
                self._weights_job_file = None
            ev = next((e for e in events if e.get('event') == 'weights'), None)
            if ev:
                self._show_weights(ev)
            else:
                self.lbl_status.setText('Weights check did not complete — see Log.')
        elif kind == 'run':
            done = next((e for e in events if e.get('event') == 'done'), None)
            if done:
                self.lbl_status.setText(f"Done in {done.get('minutes')} min. Best: "
                                        + ', '.join(f'{k}: {os.path.basename(v)}' for k, v in done['best'].items()))
                self.bar_sweep.setValue(self.bar_sweep.maximum())
            else:
                failed = next((e for e in events if e.get('event') == 'failed'), None)
                msg = '; '.join(failed['errors']) if failed else f'exit code {code} — see Log'
                self.lbl_status.setText(f'Run stopped: {msg}')

    def _on_event(self, ev):
        kind = ev.get('event')
        if kind == 'job':
            self.bar_sweep.setMaximum(max(1, len(ev.get('sweep', []))))
            self.bar_sweep.setValue(0)
        elif kind == 'sweep':
            self.bar_sweep.setValue(ev['done'])
            self.lbl_status.setText(f"Sweep point {ev['point']}")
        elif kind in ('window', 'preprocess', 'h5_read') and ev.get('total'):
            self.bar_step.setMaximum(int(ev['total']))
            self.bar_step.setValue(int(ev['done']))
            label = ev.get('matcher') or kind
            self.bar_step.setFormat(f'{label}  %v/%m')
        elif kind == 'stage':
            self.lbl_status.setText(f"{ev.get('channel', '')}: {ev['stage']} "
                                    f"{ev.get('detector', '')} {ev.get('pair', '')}".strip())
        elif kind == 'detector':
            self.lbl_status.setText(f"{ev['channel']}: running {ev['detector']}")
        elif kind == 'result':
            self._add_result(ev)
        elif kind == 'truth_result':
            self._set_truth_cell(ev)
        elif kind == 'truth':
            self._show_truth(ev)
        elif kind == 'weights_progress':
            self.bar_step.setMaximum(int(ev['total']))
            self.bar_step.setValue(int(ev['done']))
            self.bar_step.setFormat(f"weights: {ev['detector']}  %v/%m")

    def _add_result(self, ev):
        r = self.results.rowCount()
        self.results.insertRow(r)
        for c, key in enumerate(RESULT_COLS):
            v = ev.get(key)
            if isinstance(v, float):
                v = f'{v:.2f}'
            self.results.setItem(r, c, QtWidgets.QTableWidgetItem('' if v is None else str(v)))
        if ev.get('error'):
            self.results.item(r, 3).setToolTip(ev['error'])
        self.results.resizeColumnsToContents()

    def _set_truth_cell(self, ev):
        c = RESULT_COLS.index('truth_rmse_m')
        for r in range(self.results.rowCount()):
            vals = [self.results.item(r, RESULT_COLS.index(k)) for k in ('sweep', 'channel', 'detector')]
            if [v.text() if v else '' for v in vals] == [str(ev.get(k)) for k in ('sweep', 'channel', 'detector')]:
                v = ev.get('truth_rmse_m')
                it = QtWidgets.QTableWidgetItem('' if v is None else f'{v:.1f}')
                it.setToolTip(f"RMSE against the ground truth at {ev.get('truth_reached')} point(s)")
                self.results.setItem(r, c, it)
        self.results.resizeColumnsToContents()

    def _show_truth(self, ev):
        top = ev.get('top') or []
        if not top:
            self.lbl_status.setText('Ground truth: no configuration could be scored — see Log.')
            return
        dlg = TruthDialog(ev, self)
        dlg.show()
        self._last_truth_dialog = dlg
        b = top[0]
        rmse = b.get('truth_rmse_m')
        self.lbl_status.setText(f"Best vs ground truth: {b.get('detector')} + {b.get('matcher_family')} "
                                f"({b.get('best_matcher_setting')} {b.get('ransac')}), RMSE "
                                + ('-' if rmse is None else f'{rmse:.1f} m')
                                + f" at {b.get('truth_reached')}/{b.get('truth_total')} truth points")

    def _open_result(self, row, _col):
        item = self.results.item(row, RESULT_COLS.index('rival_csv'))
        path = item.text() if item else ''
        self._open_path(os.path.dirname(path) if path else self.out_path.text())

    def _show_preflight(self, ev):
        if not ev:
            self.lbl_status.setText('Preflight did not complete — see Log.')
            return
        info = ev.get('info', {})
        lines = []
        if ev['errors']:
            lines += ['ERRORS:'] + [f'  • {e}' for e in ev['errors']] + ['']
        if ev['warnings']:
            lines += ['Warnings:'] + [f'  • {w}' for w in ev['warnings']] + ['']
        refs = info.get('references', {})
        if refs:
            lines.append(f"References: {refs.get('n_footprints')} footprint(s) of "
                         f"{refs.get('n_rasters')} raster(s), mode {refs.get('mode')}")
        if 'candidate_references' in info:
            c = info['candidate_references']
            lines.append(f"Within the search buffer: {len(c)} — {', '.join(c[:8])}{' …' if len(c) > 8 else ''}")
        if 'working_resolution_m' in info:
            lines.append(f"Working resolution: {info['working_resolution_m']} m")
        if info.get('runs'):
            parts = ', '.join(f"{r['detector']}: {r['variants']} variant(s), {r['passes']} pass(es)"
                              for r in info['runs'])
            lines.append(f"Matching passes in total: {info.get('matching_passes')} ({parts}), "
                         f"each filtered with {info.get('ransac_sets_per_pass')} RANSAC setting(s)")
        if 'gpu' in info:
            lines.append(f"GPU: {info['gpu']}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Preflight ' + ('FAILED' if ev['errors'] else 'OK'))
        box.setText('\n'.join(lines) or 'All checks passed.')
        box.show()
        self.lbl_status.setText('Preflight: ' + (f"{len(ev['errors'])} error(s)" if ev['errors'] else 'OK'))
        self._last_preflight_box = box

    # ── actions ──────────────────────────────────────────────────────────────
    def _write_job(self):
        try:
            job = self.get_job()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Job', str(e))
            return None
        if not job['output_dir']:
            QtWidgets.QMessageBox.warning(self, 'Job', 'Choose an output folder.')
            return None
        os.makedirs(job['output_dir'], exist_ok=True)
        path = os.path.join(job['output_dir'], 'job_gui.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        self.settings.setValue('last_job', json.dumps(job))
        return path

    def inspect_input(self):
        if not self.in_path.text():
            QtWidgets.QMessageBox.warning(self, 'Input', 'Choose an input image first.')
            return
        self._pending_channels = [self.channels.item(i).text() for i in range(self.channels.count())
                                  if self.channels.item(i).checkState() == CHECKED]
        self._start(['inspect', self.in_path.text()], 'inspect')

    def preflight(self):
        path = self._write_job()
        if path:
            self._start(['preflight', path], 'preflight')

    def _update_kp_label(self, *_):
        """Keypoints each window size gets with 'auto', as the run computes them."""
        try:
            sizes = parse_int_list(self.windows.text(), 'window sizes')
        except ValueError:
            self.lbl_kp.setText('')
            return
        dens, cap = self.kp_density.value(), self.max_feat.value()
        parts = []
        for w in sizes:
            want = int(dens * w * w / 1e6)
            n = max(1000, min(want, cap))
            parts.append(f'{w} px: {n:,}' + (f' (capped; {want:,} at this density)' if want > cap else ''))
        self.lbl_kp.setText('auto gives ' + ' · '.join(parts) if parts else '')

    def pack_dialog(self, *_, run=True):
        """Pack the current job for the GPU server (automatch_pack) from a
        dialog; run=False returns the dialog without showing it (tests).
        run is keyword-only: a button's clicked signal passes checked=False
        positionally, which used to land in run and the dialog never opened."""
        try:
            job = self.get_job()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Pack', str(e))
            return None
        dlg = PackDialog(self.settings, job, self)
        dlg.accepted.connect(lambda: self._start_pack(dlg))
        if run:
            dlg.exec() if hasattr(dlg, 'exec') else dlg.exec_()
        return dlg

    def _start_pack(self, dlg):
        args, job = dlg.args()
        fd, path = tempfile.mkstemp(prefix='automatch_pack_', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        self._pack_job_file = path
        self._pack_dest = dlg.local.text()
        self._pack_dry = dlg.dry.isChecked()
        self._pack_lines = []
        self.log.appendPlainText(f'--- pack to {self._pack_dest}')
        self._start(['pack', path] + args, 'pack')

    def _pack_finished(self, code):
        if getattr(self, '_pack_job_file', None):
            try:
                os.remove(self._pack_job_file)
            except OSError:
                pass
            self._pack_job_file = None
        report = os.path.join(getattr(self, '_pack_dest', ''), 'PACK_REPORT.txt')
        if not self._pack_dry and os.path.exists(report):
            text = open(report, encoding='utf-8').read()
        else:
            text = '\n'.join(getattr(self, '_pack_lines', []))
        self.lbl_status.setText('Pack ' + ('listed (nothing copied)' if self._pack_dry else
                                           'finished' if code == 0 else f'stopped (exit code {code}) — see Log'))
        box = ReportDialog('Pack for the GPU server', text, self)
        box.show()
        self._last_pack_report = box

    def check_weights(self, scope='selected', load=False):
        """Run the weights check (automatch_weights) for the selected
        detectors as configured, or for every detector and weight choice."""
        args = ['weights']
        if scope == 'selected':
            try:
                job = self.get_job()
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, 'Check weights', str(e))
                return
            if not job['detectors']:
                QtWidgets.QMessageBox.information(self, 'Check weights', 'Tick at least one detector, '
                                                  'or choose "All detectors".')
                return
            fd, path = tempfile.mkstemp(prefix='automatch_weights_', suffix='.json')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(job, f, indent=2)
            self._weights_job_file = path
            args += ['--job', path]
        elif self.weights_cache.text():
            args += ['--weights-cache', self.weights_cache.text()]
        if load:
            args.append('--load')
        self.log.appendPlainText(f"--- weights check ({scope}{', load' if load else ''})")
        self._start(args, 'weights')

    def _show_weights(self, ev):
        per_det = {d['detector']: d for d in ev.get('detectors', [])}
        for r in range(self.det_table.rowCount()):
            item0 = self.det_table.item(r, 0)
            it = self.det_table.item(r, 4)
            d = per_det.get(item0.text()) if item0 else None
            if it is None or d is None:
                continue
            bad = [f for f in d.get('files', []) if f['status'] != 'ok']
            it.setText({'ok': 'ok', 'no weights': 'none needed', 'partial': 'partly checked',
                        'error': 'error'}.get(d['status'], f'{len(bad)} missing'))
            it.setToolTip('\n'.join(f"{f['status']}: {f['file']}" for f in d.get('files', []))
                          or 'no weight files')
        dlg = WeightsDialog(ev, self)
        dlg.show()
        self._last_weights_dialog = dlg
        s = ev.get('summary', {})
        self.lbl_status.setText(f"Weights: {s.get('found', 0)} of {s.get('files', 0)} file(s) found"
                                + (f", {s.get('missing')} missing" if s.get('missing') else ''))

    def run_job(self):
        path = self._write_job()
        if not path:
            return
        self.results.setRowCount(0)
        self.bar_sweep.setValue(0)
        self.bar_step.setValue(0)
        self.log.appendPlainText(f'--- run {path}')
        self._start(['run', path], 'run')

    def stop(self):
        if self.proc is None:
            return
        self.lbl_status.setText('Stopping …')
        self.proc.terminate()
        QtCore.QTimer.singleShot(5000, lambda: self.proc and self.proc.kill())

    def open_output(self):
        self._open_path(self.out_path.text())

    def _open_path(self, path):
        if path and os.path.exists(path):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def load_job_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load job', self.out_path.text(), 'Job (*.json)')
        if p:
            with open(p, encoding='utf-8') as f:
                self.set_job({**DEFAULT_JOB, **json.load(f)})

    def save_job_dialog(self):
        try:
            job = self.get_job()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Job', str(e))
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save job', self.out_path.text(), 'Job (*.json)')
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(job, f, indent=2)

    def closeEvent(self, ev):
        if self.proc is not None:
            answer = QtWidgets.QMessageBox.question(self, 'Quit', 'A job is running. Stop it and quit?')
            if answer != MSG_YES:
                ev.ignore()
                return
            self.proc.kill()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AutoMatchWindow()
    w.show()
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == '__main__':
    sys.exit(main())
