"""
gcp_train window: fine-tune LightGlue on automatch's consistent points, step
by step, and set up the check in automatch.

    python gcp_train_gui.py

Tabs, in order:
  1 Scenes   automatch output folders (any layout: server output<job>/, a
             workstation's <product>/<auto step>/ ...), the input image folders
             (searched by file name), the truth folder (matched by the input's
             file or folder name: those scenes only test), what each band is
             -> scenes_consistent.csv (consistent.py)
  2 Pairs    one labelled crop pair per accepted crop (pairs.py --labels consistent)
  3 Train    smoke test, then the fine-tune (Glue Factory), DISK or ALIKED + LightGlue
  4 Check    the official and the fine-tuned matcher on the test pairs (gcp_train
             eval), and the settings that run both side by side in automatch on
             the scenes with truth (the test that decides)

Every step is a settings file for gpuaas.py. 'Run here' runs it on this
machine; 'Save for the server' writes it with server paths (the path mapping
below) and shows the GPUaaS command. Nothing from g1a_automatch is imported.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    QT_API = 'PyQt5'
except ImportError:                                   # pragma: no cover
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        QT_API = 'PyQt6'
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets
        QT_API = 'PySide6'

HERE = os.path.dirname(os.path.abspath(__file__))
GPUAAS = os.path.join(HERE, 'gpuaas.py')
MODALITIES = ['visible', 'nir', 'swir', 'mwir', 'lwir', 'sar']


def to_server(v, local: str, server: str):
    """Paths in a settings value as the server sees them: local prefix -> server prefix, '/' separators."""
    if isinstance(v, dict):
        return {k: to_server(x, local, server) for k, x in v.items()}
    if isinstance(v, list):
        return [to_server(x, local, server) for x in v]
    if isinstance(v, str) and local and server:
        p = v.replace('\\', '/')
        lp = local.replace('\\', '/')
        if lp and p.lower().startswith(lp.lower()):
            return server.rstrip('/') + '/' + p[len(lp):].lstrip('/')
    return v


class PathRow(QtWidgets.QWidget):
    """A path field with a Browse button (a file or a folder)."""

    def __init__(self, folder: bool, caption: str, filt: str = ''):
        super().__init__()
        self.folder, self.caption, self.filt = folder, caption, filt
        self.edit = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton('Browse…')
        b.clicked.connect(self.browse)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.edit, 1)
        h.addWidget(b)

    def browse(self):
        start = self.edit.text() or ''
        if self.folder:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, self.caption, start)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.caption, start, self.filt)
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t or '')


class FolderList(QtWidgets.QWidget):
    """Several folders: a list with Add and Remove."""

    def __init__(self, caption: str):
        super().__init__()
        self.caption = caption
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(70)
        add, rem = QtWidgets.QPushButton('Add…'), QtWidgets.QPushButton('Remove')
        add.clicked.connect(self.add)
        rem.clicked.connect(lambda: [self.list.takeItem(self.list.row(i)) for i in self.list.selectedItems()])
        v = QtWidgets.QVBoxLayout()
        v.addWidget(add)
        v.addWidget(rem)
        v.addStretch(1)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.list, 1)
        h.addLayout(v)

    def add(self, path: str = ''):
        p = path or QtWidgets.QFileDialog.getExistingDirectory(self, self.caption)
        if p and p not in self.paths():
            self.list.addItem(p)

    def paths(self) -> List[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def set_paths(self, paths: List[str]):
        self.list.clear()
        for p in paths or []:
            self.list.addItem(p)


def _spin(lo, hi, val, step=None, decimals=None):
    s = QtWidgets.QDoubleSpinBox() if decimals is not None else QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    if decimals is not None:
        s.setDecimals(decimals)
    if step:
        s.setSingleStep(step)
    s.setValue(val)
    return s


class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle('gcp_train: fine-tune LightGlue on automatch\'s consistent points')
        self.settings = settings or QtCore.QSettings('DPQED', 'gcp_train')
        self.proc = None
        self._step = ''
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        v = QtWidgets.QVBoxLayout(w)
        # where the work goes, and how the server sees this machine's paths
        top = QtWidgets.QFormLayout()
        self.work = PathRow(True, 'Work folder')
        self.work.edit.setToolTip('Each step\'s settings and results go here: scenes/, pairs/, train_<name>/, eval/')
        top.addRow('Work folder', self.work)
        hm = QtWidgets.QHBoxLayout()
        self.map_local = QtWidgets.QLineEdit('V:\\')
        self.map_server = QtWidgets.QLineEdit('/maintenance/')
        hm.addWidget(self.map_local)
        hm.addWidget(QtWidgets.QLabel('here  =  on the server'))
        hm.addWidget(self.map_server)
        top.addRow('Path mapping', hm)
        v.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs, 1)
        self._tab_scenes()
        self._tab_pairs()
        self._tab_train()
        self._tab_check()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        hb = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel('')
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        hb.addWidget(self.status, 1)
        hb.addWidget(self.btn_stop)
        v.addLayout(hb)
        v.addWidget(self.log, 1)
        self._restore()
        self.resize(1000, 820)

    # ── tabs ─────────────────────────────────────────────────────────────────
    def _buttons(self, step: str, run_label: str = 'Run here') -> QtWidgets.QHBoxLayout:
        h = QtWidgets.QHBoxLayout()
        run = QtWidgets.QPushButton(run_label)
        run.clicked.connect(lambda: self.run_here(step))
        srv = QtWidgets.QPushButton('Save for the server…')
        srv.setToolTip('Write this step\'s settings with server paths, and copy the GPUaaS command')
        srv.clicked.connect(lambda: self.save_for_server(step))
        h.addWidget(run)
        h.addWidget(srv)
        h.addStretch(1)
        return h

    def _tab_scenes(self):
        t = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(t)
        self.outputs = FolderList('automatch output folder')
        self.outputs.setToolTip('Each folder of automatch job outputs, any layout: job folders (with job_used.json) '
                                'one or two levels below, e.g. output/output1082 or <product>/auto_top3_agree2')
        f.addRow('automatch outputs', self.outputs)
        self.inputs = FolderList('Input image folder')
        self.inputs.setToolTip('Where the input images are, any layout: searched by file name for a job whose '
                               'saved input path is from another machine')
        f.addRow('Input image folders', self.inputs)
        self.truth = PathRow(True, 'Truth folder')
        self.truth.edit.setToolTip('Truth files named after the input: its file name or one of its folders (e.g. '
                                   'the request ID); pts_<name>_band4.csv also matches. Those scenes only test')
        f.addRow('Truth folder', self.truth)
        self.band_mod = QtWidgets.QLineEdit('band4=nir')
        self.band_mod.setToolTip("What each band is: 'band4=nir,band3=visible' or per sensor 'G1A_MV:band4=nir' "
                                 f"({', '.join(MODALITIES)}). Not given: visible")
        f.addRow('Band modality', self.band_mod)
        h = QtWidgets.QHBoxLayout()
        self.level = _spin(0, 1024, 0)
        self.level.setSpecialValueText('finest')
        self.unreliable = QtWidgets.QCheckBox('also unreliable cross-checks')
        h.addWidget(QtWidgets.QLabel('Agreement level (px)'))
        h.addWidget(self.level)
        h.addWidget(self.unreliable)
        h.addStretch(1)
        f.addRow(h)
        f.addRow(self._buttons('consistent', 'Build the scene list here'))
        self.scene_table = QtWidgets.QTableWidget()
        f.addRow(self.scene_table)
        self.tabs.addTab(t, '1 Scenes')

    def _tab_pairs(self):
        t = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(t)
        h = QtWidgets.QHBoxLayout()
        self.crop = _spin(64, 2048, 512, 64)
        self.per_scene = _spin(1, 1000, 60)
        for lab, wd in (('Crop (px)', self.crop), ('Crops per scene', self.per_scene)):
            h.addWidget(QtWidgets.QLabel(lab))
            h.addWidget(wd)
        h.addStretch(1)
        f.addRow(h)
        h2 = QtWidgets.QHBoxLayout()
        self.min_points = _spin(4, 1000, 20)
        self.max_rms = _spin(0.1, 10.0, 1.5, 0.1, 2)
        self.max_out = _spin(0.0, 0.5, 0.1, 0.01, 2)
        self.min_cover = _spin(0.0, 1.0, 0.5, 0.05, 2)
        self.min_fam = _spin(1, 10, 2)
        self.min_agree = _spin(2, 20, 3)
        for lab, wd in (('points ≥', self.min_points), ('fit ≤ px', self.max_rms), ('outliers ≤', self.max_out),
                        ('cover ≥', self.min_cover), ('families ≥', self.min_fam), ('agreed by ≥', self.min_agree)):
            h2.addWidget(QtWidgets.QLabel(lab))
            h2.addWidget(wd)
        h2.addStretch(1)
        f.addRow('A crop is labelled when', h2)
        self.manifest = PathRow(False, 'Scene list', 'CSV (*.csv)')
        self.manifest.edit.setPlaceholderText('default: <work>/scenes/scenes_consistent.csv')
        f.addRow('Scene list', self.manifest)
        f.addRow(self._buttons('pairs', 'Build the pairs here'))
        self.fit_table = QtWidgets.QTableWidget()
        f.addRow(self.fit_table)
        self.tabs.addTab(t, '2 Pairs')

    def _tab_train(self):
        t = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(t)
        h = QtWidgets.QHBoxLayout()
        self.config = QtWidgets.QComboBox()
        self.config.addItems(['disk', 'aliked'])
        self.experiment = QtWidgets.QLineEdit('g1a_mv_disk')
        self.epochs = _spin(1, 500, 20)
        self.batch = _spin(1, 128, 16)
        for lab, wd in (('Keypoints', self.config), ('Name', self.experiment), ('Epochs', self.epochs),
                        ('Batch', self.batch)):
            h.addWidget(QtWidgets.QLabel(lab))
            h.addWidget(wd)
        h.addStretch(1)
        f.addRow(h)
        h2 = QtWidgets.QHBoxLayout()
        self.sensors = QtWidgets.QLineEdit('G1A_MV')
        self.sensors.setToolTip('Train on these sensors only (comma-separated); empty: all in the pairs')
        self.condition = QtWidgets.QCheckBox('tell the matcher modality and pixel size (conditioning)')
        self.condition.setChecked(True)
        h2.addWidget(QtWidgets.QLabel('Sensors'))
        h2.addWidget(self.sensors)
        h2.addWidget(self.condition)
        h2.addStretch(1)
        f.addRow(h2)
        self.index = PathRow(False, 'pairs.csv', 'CSV (*.csv)')
        self.index.edit.setPlaceholderText('default: <work>/pairs/pairs/pairs.csv')
        f.addRow('Pairs', self.index)
        hs = self._buttons('smoke', 'Smoke test here (minutes)')
        f.addRow('First', hs)
        f.addRow('Then', self._buttons('train', 'Train here (hours)'))
        note = QtWidgets.QLabel('Training needs Glue Factory in this Python (pip install -e glue-factory) and the '
                                'DISK / ALIKED and LightGlue weights in the torch hub cache. The keypoint network '
                                'stays as it is (automatch cannot load a retrained one yet).')
        note.setWordWrap(True)
        f.addRow(note)
        self.tabs.addTab(t, '3 Train')

    def _tab_check(self):
        t = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(t)
        self.weights = PathRow(False, 'Fine-tuned LightGlue', 'PyTorch (*.pth)')
        self.weights.edit.setPlaceholderText('default: <work>/train_<name>/<name>_lightglue.pth')
        f.addRow('Fine-tuned weights', self.weights)
        f.addRow('On the test pairs', self._buttons('eval', 'Compare with the official weights here'))
        h = QtWidgets.QHBoxLayout()
        self.mod_in = QtWidgets.QComboBox()
        self.mod_in.addItems(MODALITIES)
        self.mod_in.setCurrentText('nir')
        self.mod_ref = QtWidgets.QComboBox()
        self.mod_ref.addItems(MODALITIES)
        for lab, wd in (('input is', self.mod_in), ('reference is', self.mod_ref)):
            h.addWidget(QtWidgets.QLabel(lab))
            h.addWidget(wd)
        h.addStretch(1)
        f.addRow('In automatch', h)
        b = QtWidgets.QPushButton('Write the automatch settings…')
        b.setToolTip('The settings to add to an automatch job: DISK + LightGlue with the official and the '
                     'fine-tuned weights side by side; run it on the scenes with truth and compare lgm with '
                     'lgm_lwt<name> in TRUTH_RANKING.csv')
        b.clicked.connect(self.write_automatch)
        f.addRow(b)
        self.automatch_text = QtWidgets.QPlainTextEdit()
        self.automatch_text.setReadOnly(True)
        self.automatch_text.setMaximumHeight(150)
        f.addRow(self.automatch_text)
        self.tabs.addTab(t, '4 Check')

    # ── settings per step ────────────────────────────────────────────────────
    def work_dir(self) -> str:
        return self.work.text() or os.path.join(os.path.expanduser('~'), 'gcp_train_work')

    def out_dir(self, step: str) -> str:
        name = self.experiment.text().strip() or 'finetune'
        sub = {'consistent': 'scenes', 'pairs': 'pairs', 'smoke': f'smoke_{name}', 'train': f'train_{name}',
               'eval': 'eval', 'eval_official': 'eval'}[step]
        return os.path.join(self.work_dir(), sub)

    def manifest_path(self) -> str:
        return self.manifest.text() or os.path.join(self.out_dir('consistent'), 'scenes_consistent.csv')

    def index_path(self) -> str:
        return self.index.text() or os.path.join(self.out_dir('pairs'), 'pairs', 'pairs.csv')

    def weights_path(self) -> str:
        name = self.experiment.text().strip() or 'finetune'
        return self.weights.text() or os.path.join(self.out_dir('train'), f'{name}_lightglue.pth')

    def step_settings(self, step: str) -> Dict:
        """The settings file of one step (gpuaas.py)."""
        if step == 'consistent':
            s = {'mode': 'consistent', 'jobs': self.outputs.paths(), 'inputs': self.inputs.paths(),
                 'truth_folder': self.truth.text(), 'band_modality': self.band_mod.text().strip()}
            if self.level.value():
                s['level'] = self.level.value()
            if self.unreliable.isChecked():
                s['include_unreliable'] = True
            return s
        if step == 'pairs':
            return {'mode': 'pairs', 'manifest': self.manifest_path(), 'labels': 'consistent',
                    'crop': self.crop.value(), 'per_scene': self.per_scene.value(),
                    'local': {'min_points': self.min_points.value(), 'max_rms_px': round(self.max_rms.value(), 2),
                              'max_outliers': round(self.max_out.value(), 2),
                              'min_cover': round(self.min_cover.value(), 2), 'min_families': self.min_fam.value(),
                              'min_agree': self.min_agree.value()}}
        if step in ('smoke', 'train'):
            s = {'mode': 'train', 'index': self.index_path(), 'config': self.config.currentText(),
                 'experiment': self.experiment.text().strip() or 'finetune'}
            sens = [x.strip() for x in self.sensors.text().split(',') if x.strip()]
            if sens:
                s['sensors'] = sens
            if not self.condition.isChecked():
                s['condition'] = False
            if step == 'smoke':
                s['smoke'] = True
            else:
                s.update(epochs=self.epochs.value(), batch_size=self.batch.value())
            return s
        if step == 'eval':                       # the fine-tuned weights (eval_official: the official ones)
            return {'mode': 'eval', 'index': self.index_path(), 'config': self.config.currentText(),
                    'checkpoint': os.path.join(self.out_dir('train'), 'checkpoint_best.tar'),
                    'label': self.experiment.text().strip() or 'finetune'}
        if step == 'eval_official':
            return {'mode': 'eval', 'index': self.index_path(), 'config': self.config.currentText(),
                    'label': 'official'}
        raise ValueError(step)

    def check_step(self, step: str) -> str:
        """Why a step cannot run yet ('' when it can)."""
        s = self.step_settings(step)
        if step == 'consistent' and not s['jobs']:
            return 'Add at least one automatch output folder.'
        if step == 'pairs' and not os.path.exists(s['manifest']):
            return f'No scene list yet ({s["manifest"]}): build it in 1 Scenes.'
        if step in ('smoke', 'train', 'eval') and not os.path.exists(s['index']):
            return f'No pairs yet ({s["index"]}): build them in 2 Pairs.'
        if step == 'eval' and not os.path.exists(s['checkpoint']):
            return f'No fine-tuned model yet ({s["checkpoint"]}): train it in 3 Train.'
        return ''

    def write_settings(self, step: str, server: bool = False) -> str:
        s = self.step_settings(step)
        folder = os.path.join(self.work_dir(), 'server' if server else 'settings')
        if server:
            s = to_server(s, self.map_local.text().strip(), self.map_server.text().strip())
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, f'{step}.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2)
        return p

    # ── running ──────────────────────────────────────────────────────────────
    def run_here(self, step: str):
        if self.proc is not None:
            return
        why = self.check_step(step)
        if why:
            QtWidgets.QMessageBox.warning(self, 'gcp_train', why)
            return
        self._save()
        if step == 'eval':                       # the official weights first, then the fine-tuned ones
            self._chain = ['eval']
            step = 'eval_official'
        self._start_step(step)

    def _start_step(self, step: str):
        path = self.write_settings(step)
        out = self.out_dir(step)
        os.makedirs(out, exist_ok=True)
        self.log.appendPlainText(f'--- {step}: {path} -> {out}')
        self._step = step
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels if QT_API != 'PyQt5'
                                        else QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.btn_stop.setEnabled(True)
        self.status.setText(f'{step}: running …')
        self.proc.start(sys.executable, ['-u', GPUAAS, '.', out, path])

    def _read(self):
        txt = bytes(self.proc.readAllStandardOutput()).decode('utf-8', 'replace')
        for line in txt.splitlines():
            self.log.appendPlainText(line)

    def _finished(self, code, *_):
        step, self.proc = self._step, None
        chain = getattr(self, '_chain', [])
        if code == 0 and chain:
            self._chain = chain[1:]
            self._start_step(chain[0])
            return
        self._chain = []
        self.btn_stop.setEnabled(False)
        self.status.setText(f'{step}: ' + ('done' if code == 0 else f'ended with code {code} (see the log)'))
        self.after_step(step)

    def after_step(self, step: str):
        """Show what a step made."""
        if step == 'consistent' and os.path.exists(self.manifest_path()):
            self.fill_table(self.scene_table, self.manifest_path(),
                            ['scene_id', 'sensor', 'split', 'modality', 'points', 'truth_csv', 'job'])
            self.tabs.setCurrentIndex(1)
        if step == 'pairs':
            fit = os.path.join(self.out_dir('pairs'), 'pairs', 'scenes_fit.csv')
            if os.path.exists(fit):
                self.fill_table(self.fit_table, fit, ['scene_id', 'split', 'pairs', 'label_rms_px_median',
                                                      'crops_refused', 'why'])
        if step == 'train' and os.path.exists(self.weights_path()):
            self.weights.setText(self.weights_path())
            self.tabs.setCurrentIndex(3)
        if step == 'eval':
            import glob as _g
            files = sorted(_g.glob(os.path.join(self.out_dir('eval'), 'EVAL_*.csv')))
            if files:
                self.log.appendPlainText('--- on the test pairs: ' + ', '.join(os.path.basename(f) for f in files)
                                         + ' (matches per pair, precision, homography success; overall and per '
                                           'sensor, modality and pixel size)')

    def fill_table(self, table: QtWidgets.QTableWidget, csv_path: str, cols: List[str]):
        import pandas as pd
        try:
            d = pd.read_csv(csv_path)
        except (OSError, ValueError):
            return
        cols = [c for c in cols if c in d.columns]
        table.setColumnCount(len(cols))
        table.setRowCount(len(d))
        table.setHorizontalHeaderLabels(cols)
        for i, r in enumerate(d[cols].fillna('').itertuples(index=False)):
            for k, v in enumerate(r):
                table.setItem(i, k, QtWidgets.QTableWidgetItem(os.path.basename(str(v)) if cols[k] == 'truth_csv'
                                                               else str(v)))
        table.resizeColumnsToContents()

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.log.appendPlainText('--- stopped')

    def save_for_server(self, step: str):
        self._save()
        ms, ss = self.map_local.text().strip(), self.map_server.text().strip()
        cmds = []
        for st in (['eval_official', 'eval'] if step == 'eval' else [step]):
            p = self.write_settings(st, server=True)
            cmds.append(f'python gcp_train/gpuaas.py "." {to_server(self.out_dir(st), ms, ss)} {to_server(p, ms, ss)}')
        cmd = '\n'.join(cmds)
        QtWidgets.QApplication.clipboard().setText(cmd)
        self.log.appendPlainText(f'--- {step} for the server:\n    ' + cmd.replace('\n', '\n    ') + '   (copied)')
        local_left = [x for x in json.dumps(json.load(open(p, encoding='utf-8'))).split('"')
                      if re.match(r'^[A-Za-z]:[\\/]', x)]
        if local_left:
            self.log.appendPlainText('    WARNING: paths the server cannot see (outside the mapping): '
                                     + ', '.join(sorted(set(local_left))[:5]))
        self.status.setText(f'{step}: settings for the server written; the command is in the clipboard')

    def automatch_settings(self) -> Dict:
        name = re.sub(r'[^A-Za-z0-9_]', '_', self.experiment.text().strip() or 'finetune')
        feat = self.config.currentText()
        w = to_server(self.weights_path(), self.map_local.text().strip(), self.map_server.text().strip())
        return {'lightglue_weights': {name: {'feature': feat, 'weights': w,
                                             'modality': [self.mod_in.currentText(), self.mod_ref.currentText()]}},
                'detectors': [feat], 'matchers': {feat: ['lgm']},
                'detector_params': {feat: {'lgm.weights': ['official', name]}}}

    def write_automatch(self):
        s = self.automatch_settings()
        os.makedirs(self.work_dir(), exist_ok=True)
        p = os.path.join(self.work_dir(), 'automatch_lightglue.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2)
        name = next(iter(s['lightglue_weights']))
        self.automatch_text.setPlainText(
            json.dumps(s, indent=2) + f'\n\n# {p}\n# add to the automatch job of each scene with truth (its truth '
            f'file set); in TRUTH_RANKING.csv compare lgm (official) with lgm_lwt{re.sub("[^A-Za-z0-9]", "", name)[:16]} '
            f'(fine-tuned). Keep it only with more correct matches, an equal or better truth error and no scene worse.')
        self.log.appendPlainText(f'--- automatch settings -> {p}')

    # ── remembered between sessions ──────────────────────────────────────────
    def _save(self):
        st = self.settings
        st.setValue('work', self.work.text())
        st.setValue('map_local', self.map_local.text())
        st.setValue('map_server', self.map_server.text())
        st.setValue('outputs', json.dumps(self.outputs.paths()))
        st.setValue('inputs', json.dumps(self.inputs.paths()))
        st.setValue('truth', self.truth.text())
        st.setValue('band_mod', self.band_mod.text())
        st.setValue('experiment', self.experiment.text())
        st.setValue('config', self.config.currentText())

    def _restore(self):
        st = self.settings
        self.work.setText(st.value('work', '') or '')
        self.map_local.setText(st.value('map_local', 'V:\\') or 'V:\\')
        self.map_server.setText(st.value('map_server', '/maintenance/') or '/maintenance/')
        try:
            self.outputs.set_paths(json.loads(st.value('outputs', '[]') or '[]'))
            self.inputs.set_paths(json.loads(st.value('inputs', '[]') or '[]'))
        except ValueError:
            pass
        self.truth.setText(st.value('truth', '') or '')
        self.band_mod.setText(st.value('band_mod', 'band4=nir') or '')
        self.experiment.setText(st.value('experiment', 'g1a_mv_disk') or 'g1a_mv_disk')
        self.config.setCurrentText(st.value('config', 'disk') or 'disk')

    def closeEvent(self, e):
        self._save()
        super().closeEvent(e)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = TrainWindow()
    w.show()
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == '__main__':
    sys.exit(main())
