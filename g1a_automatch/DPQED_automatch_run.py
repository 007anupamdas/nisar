"""
G1A geolocation matching, the short way: pick the image, the reference and
the output folder, and run with the default settings of its sensor
(automatch_presets: MV / HV / HS / ML, from the benchmark). The result to
load in RIVAL is RIVAL_FINAL_<band>.csv in the output folder.

    python DPQED_automatch_run.py

DPQED_automatch.py is the full window (every setting, sweeps, benchmark);
this one runs the settings found there.
"""
from __future__ import annotations

import json
import os
import sys

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
import automatch_presets as P  # noqa: E402
from automatch_job import AUTOMATCH_VERSION, PROGRESS_PREFIX  # noqa: E402  (no torch import)


def kill_tree(proc) -> None:
    """End a QProcess and every process it started (the GPU workers of a
    two-GPU run): taskkill /T on Windows, the children first elsewhere. The
    workers also end themselves when their parent is gone."""
    import subprocess
    try:
        pid = int(proc.processId())
    except Exception:
        pid = 0
    if pid > 0:
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, timeout=15,
                               creationflags=0x08000000)
            else:
                subprocess.run(['pkill', '-TERM', '-P', str(pid)], capture_output=True, timeout=15)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def job_summary(out_dir: str, minutes=None) -> str:
    """What a finished job left, in a few lines, for the job-complete dialog."""
    import glob as _glob
    lines = [f'Output: {out_dir}'] + ([f'Done in {float(minutes):.1f} min'] if minutes not in (None, '') else [])
    try:
        import pandas as pd
        c = pd.read_csv(os.path.join(out_dir, 'CROSS_SUMMARY.csv'))
        for r in c.to_dict('records'):
            ch = r.get('channel')
            lines.append(f"{ch}: cross-check {'reliable' if str(r.get('reliable')).lower() == 'true' else 'NOT reliable'}"
                         f", {r.get('points')} agreed points")
            for k in sorted(k for k in r if k.startswith('final_points_')):
                lv = k[len('final_points_'):]
                share = r.get(f'rigid_share_{lv}')
                lines.append(f"  {lv}: {r[k]} final points ({r.get(f'final_tiles_ok_{lv}')} of "
                             f"{r.get(f'final_tiles_{lv}')} tiles)"
                             + (f", {100 * float(share):.0f} % fit one offset" if share == share and share is not None
                                else ''))
    except Exception:
        pass
    try:
        import pandas as pd
        st = pd.read_csv(os.path.join(out_dir, 'REPORT', 'REPORT_STATS.csv'))
        man = st[st['file'].astype(str).str.startswith('MANUAL')]
        for r in st[st['file'].astype(str).str.startswith('RIVAL_FINAL_')].to_dict('records'):
            lines.append(f"{r['file']}: mean dE {float(r.get('mean_dx_m') or 0) / 1000:.3f} km, dN "
                         f"{float(r.get('mean_dy_m') or 0) / 1000:.3f} km, CE90 {float(r.get('ce90_m') or 0) / 1000:.3f} km"
                         + (f"  (manual CE90 {float(man['ce90_m'].iloc[0]) / 1000:.3f} km)" if len(man) else ''))
    except Exception:
        pass
    auto = os.path.join(os.path.dirname(os.path.normpath(out_dir)), 'AUTO_PRESET.txt')
    if os.path.exists(auto):
        lines.append('Auto: ' + open(auto, encoding='utf-8').read().strip().replace('\n', '; '))
    n_png = len(_glob.glob(os.path.join(out_dir, 'REPORT', '*_vectors.png')))
    if n_png:
        lines.append(f'{n_png} look-vector plot(s) in REPORT/')
    return '\n'.join(lines)


def show_job_done(parent, out_dir: str, minutes=None):
    """A non-blocking job-complete dialog with the summary and buttons to the folder and the plots."""
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle('Job complete')
    box.setText(job_summary(out_dir, minutes))
    open_b = box.addButton('Open folder', QtWidgets.QMessageBox.ButtonRole.ActionRole
                           if QT_API != 'PyQt5' else QtWidgets.QMessageBox.ActionRole)
    plots_b = box.addButton('Open plots', QtWidgets.QMessageBox.ButtonRole.ActionRole
                            if QT_API != 'PyQt5' else QtWidgets.QMessageBox.ActionRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Close if QT_API != 'PyQt5' else QtWidgets.QMessageBox.Close)

    def clicked(b):
        target = out_dir if b is open_b else os.path.join(out_dir, 'REPORT') if b is plots_b else ''
        if target and os.path.isdir(target):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))
    box.buttonClicked.connect(clicked)
    box.setModal(False)
    box.show()
    return box

DEVICES = ('all', 'auto', 'cuda:0', 'cuda:1', 'cpu')


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


class RunWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle(f'G1A automatch: run with sensor defaults ({AUTOMATCH_VERSION})')
        self.settings = settings or QtCore.QSettings('DPQED', 'automatch_run')
        self.presets = P.load()
        self.proc = None
        self._kind = ''
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        v = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        v.addLayout(f)
        self.input = PathRow(False, 'Input image', 'Images (*.tif *.tiff *.h5);;All files (*)')
        self.reference = PathRow(True, 'Reference folder (L8)')
        self.reference_c1 = PathRow(True, 'C1 reference folder (Indian region)')
        self.output = PathRow(True, 'Output folder')
        self.truth = PathRow(False, 'Ground truth (optional)', 'CSV (*.csv);;All files (*)')
        self.weights = PathRow(True, 'Weights folder (optional)')
        f.addRow('Input image', self.input)
        f.addRow('Reference folder (L8)', self.reference)
        f.addRow('C1 references (India)', self.reference_c1)
        self.reference_c1.setToolTip('Optional: with a scene centre in the Indian region (68-98 E, 6-38 N) the job '
                                     'uses these references instead of the L8 ones')
        f.addRow('Output folder', self.output)
        f.addRow('Ground truth (optional)', self.truth)
        f.addRow('Weights folder (optional)', self.weights)
        self.sensor = QtWidgets.QComboBox()
        self.sensor.addItems(['from the file name'] + list(self.presets))
        self.sensor.setToolTip('The preset used. From the file name: G1A_MV_ / G1A_HV_ / G1A_HS_ / G1A_ML_')
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(list(P.MODES))
        self.mode.setToolTip('accurate: the sensor\'s ranked detectors, MAGSAC + LMedS, 512 px sub-cells, 3 of 5 '
                             'agreeing. fastest: its fast detectors (1-3 min per pass), 2 of 3 '
                             'agreeing: about a fifth of the time')
        self.mode.setToolTip(self.mode.toolTip() + '. most accurate: the ranked list plus the slow detectors that '
                             'ranked well (DKM, RoMa, KeyNet, OmniGlue, ...), all run, 4 agreeing: several times '
                             'the time')
        self.mode.setCurrentText(self.settings.value('mode', 'accurate') or 'accurate')
        self.mode.currentIndexChanged.connect(lambda *_: (setattr(self, '_shown_sensor', None), self.refresh()))
        self.bands = QtWidgets.QLineEdit()
        self.bands.setPlaceholderText('e.g. 28 or 28, 50')
        self.bands.setMaximumWidth(160)
        self.bands.setToolTip('The band(s) to match: yours to name. Empty is accepted only with crispest ticked, '
                              'or with a truth file whose name ends in the band (pts_band28.csv)')
        self.crispest = QtWidgets.QCheckBox('crispest')
        self.crispest.setToolTip('Instead of naming a band: let the job pick the band with the most fine detail '
                                 '(hyperspectral cubes). Off by default')
        self.crispest.toggled.connect(lambda on: self.bands.setEnabled(not on))
        self.max_err = QtWidgets.QDoubleSpinBox()
        self.max_err.setRange(1.0, 500.0)
        self.max_err.setValue(25.0)
        self.max_err.setSuffix(' km')
        self.max_err.setToolTip('Largest geolocation error expected. With a ground-truth file it is taken from the '
                                'file instead')
        self.device = QtWidgets.QComboBox()
        self.device.addItems(DEVICES)
        self.device.setToolTip('all: every GPU of this machine, sharing one queue of the detectors')
        self.top_n = QtWidgets.QSpinBox()
        self.top_n.setRange(1, 40)
        self.top_n.setToolTip('How many of the sensor\'s ranked detectors run (the first N of its list)')
        self.agree = QtWidgets.QSpinBox()
        self.agree.setRange(1, 40)
        self.agree.setToolTip('The error is taken where at least this many of them agree (cross-check); '
                              'the points they agree on go to RIVAL_FINAL')
        self.final_points = QtWidgets.QSpinBox()
        self.final_points.setRange(0, 1000)
        self.final_points.setValue(24)
        self.final_points.setToolTip('Refined points for RIVAL (RIVAL_FINAL): one best agreed point per sub-tile of a '
                                     'grid of about this many over the whole scene; tiles on water, cloud or '
                                     'featureless ground fail (FINAL_TILES_*.csv says why)')
        self.cell_px = QtWidgets.QLineEdit('64, 128')
        self.cell_px.setMaximumWidth(90)
        self.cell_px.setToolTip('Agreement cells: in cells of this many pixels (8 x 8 in a 512 px window) a point '
                                'counts only where the top detectors agree (error from top Y), the best keypoint of '
                                'each is kept; the refined points are picked among these, the most agreeing and '
                                'tightest first. 0: off')
        self.cloud = QtWidgets.QCheckBox('mask clouds')
        self.cloud.setChecked(True)
        self.cloud.setToolTip('By the sensor: MV OmniCloudMask on bands 5, 4, 6 (needs it installed); HV VNIR, '
                              'HS SWIR, ML infrared tests')
        h = QtWidgets.QHBoxLayout()
        for label, wid in (('Sensor', self.sensor), ('Preset', self.mode), ('Band(s)', self.bands), ('Max error (no truth)', self.max_err),
                           ('Detectors: top', self.top_n), ('error from top', self.agree),
                           ('Agreement cells', self.cell_px), ('Refined points', self.final_points),
                           ('Device', self.device)):
            h.addWidget(QtWidgets.QLabel(label))
            h.addWidget(wid)
            if wid is self.bands:
                h.addWidget(self.crispest)
        h.addWidget(self.cloud)
        h.addStretch(1)
        v.addLayout(h)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel if QT_API != 'PyQt5'
                                   else QtWidgets.QFrame.StyledPanel)
        self.summary.setMargin(6)
        v.addWidget(self.summary)
        hb = QtWidgets.QHBoxLayout()
        self.btn_check = QtWidgets.QPushButton('Check')
        self.btn_run = QtWidgets.QPushButton('Run')
        self.btn_all = QtWidgets.QPushButton('Run all 3 presets')
        self.btn_auto = QtWidgets.QPushButton('Run auto (by distortion)')
        self.btn_auto.setToolTip('The fastest preset first; then by how much of the scene fits one offset (rigid share): '
                                 'over 80 % stop there (little distortion), 40-80 % accurate, under 40 % most accurate. '
                                 'One cache, so the second run only adds detectors. The choice is written to '
                                 'AUTO_PRESET.txt in the output folder')
        self.btn_all.setToolTip('fastest, then accurate, then most accurate, into <output>/fastest, /accurate, '
                                '/most_accurate, sharing one cache (<output>/_cache): references, pairs and the cloud '
                                'mask are made once, and a detector one preset ran is reused by the next')
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_open = QtWidgets.QPushButton('Open output folder')
        self.btn_presets = QtWidgets.QPushButton('Presets file…')
        self.btn_presets.setToolTip('Write automatch_presets.json (the presets, to edit) or show where it is; '
                                    'it is read when this window opens')
        self.btn_stop.setEnabled(False)
        for b in (self.btn_check, self.btn_run, self.btn_all, self.btn_auto, self.btn_stop, self.btn_open, self.btn_presets):
            hb.addWidget(b)
        hb.addStretch(1)
        v.addLayout(hb)
        self.bar = QtWidgets.QProgressBar()
        self.stage = QtWidgets.QLabel('')
        v.addWidget(self.bar)
        self.gpu_bars = [QtWidgets.QProgressBar(), QtWidgets.QProgressBar()]   # one per GPU of a two-GPU run
        for b in self.gpu_bars:
            b.setVisible(False)
            v.addWidget(b)
        self._bar_of, self._det_done, self._det_total = {}, 0, 0
        v.addWidget(self.stage)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        v.addWidget(self.log, 1)
        self.btn_check.clicked.connect(lambda: self.start('preflight'))
        self.btn_run.clicked.connect(lambda: self.start('run'))
        self.btn_all.clicked.connect(self.run_all)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open.clicked.connect(self.open_output)
        self.btn_presets.clicked.connect(self.presets_file)
        self.input.edit.textChanged.connect(self.refresh)
        self.sensor.currentIndexChanged.connect(self.refresh)
        for key, row in (('input', self.input), ('reference', self.reference), ('reference_c1', self.reference_c1),
                         ('output', self.output), ('truth', self.truth), ('weights', self.weights)):
            row.edit.setText(self.settings.value(key, '') or '')
        self.device.setCurrentText(self.settings.value('device', 'all') or 'all')
        self.cloud.setChecked(str(self.settings.value('cloud', 'true')).lower() == 'true')
        try:
            self.max_err.setValue(float(self.settings.value('max_err', 25.0)))
        except (TypeError, ValueError):
            pass
        self.resize(980, 720)
        self.refresh()

    # ── the preset in use ────────────────────────────────────────────────────
    def current_sensor(self) -> str:
        return self.sensor.currentText() if self.sensor.currentIndex() > 0 else P.sensor_of(self.input.text())

    def refresh(self, *_):
        s = self.current_sensor()
        if s in self.presets:
            m = self.mode.currentText()
            p = dict(self.presets[s], **P.MODES[m])
            if m == 'fastest' and p.get('fast_detectors'):
                p['detectors'] = p['fast_detectors']
            elif m == 'most accurate' and p.get('slow_detectors'):
                p['detectors'] = list(p['detectors']) + [d for d in p['slow_detectors'] if d not in p['detectors']]
            if s != getattr(self, '_shown_sensor', None):     # the preset's own numbers, until changed here
                self.top_n.setValue(int(p.get('top_n', 5) or 5))
                self.agree.setValue(int(p.get('agree', 3) or 3))
                self._shown_sensor = s
            self.summary.setText(f'<b>{s} preset, {m}</b> ({P.BASIS})<br>' + P.describe(p).replace('\n', '<br>'))
        else:
            self.summary.setText('Sensor not known from the file name (G1A_MV_ / HV / HS / ML): choose it under Sensor.')

    def job(self, mode: str = '', sub: str = ''):
        s = self.current_sensor()
        if s not in self.presets:
            raise ValueError('choose the sensor (the file name does not say it)')
        for label, row in (('input image', self.input), ('reference folder', self.reference),
                           ('output folder', self.output)):
            if not row.text():
                raise ValueError(f'set the {label}')
        crisp = self.crispest.isChecked()
        ref, why = P.reference_for(self.input.text(), self.reference.text(), self.reference_c1.text())
        self.log.appendPlainText(f'--- references: {why}: {ref}')
        out = os.path.join(self.output.text(), sub) if sub else self.output.text()
        job, note = P.job_for(s, self.input.text(), ref, out, self.truth.text(),
                           [] if crisp else P.parse_bands(self.bands.text()),
                           None if self.truth.text() else self.max_err.value(),
                           self.device.currentText(), self.cloud.isChecked(), self.weights.text(), self.presets,
                           crispest=1 if crisp else 0,
                           top_n=None if mode else self.top_n.value(), agree=None if mode else self.agree.value(),
                           mode=mode or self.mode.currentText(), final_points=self.final_points.value(),
                           cell_px=P.parse_levels(self.cell_px.text()),
                           temp_dir=os.path.join(self.output.text(), '_cache'))
        if '[window' in note:
            self.log.appendPlainText('--- ' + note[note.index('[window') + 1:].rstrip(']'))
        return job

    def run_auto(self):
        """The fastest preset, then the one its rigid share calls for (P.auto_next)."""
        if self.proc is not None:
            return
        try:
            job = self.job('fastest', 'fastest')
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Automatch', str(e))
            return
        self._pending, self._all_done = [], []
        self._auto = True
        self.start('run', ('fastest', job))

    def run_all(self):
        """The three presets one after the other (their own top X / Y), each
        into its own subfolder, one cache: the later ones reuse what the
        earlier ones made."""
        if self.proc is not None:
            return
        try:
            jobs = [(m, self.job(m, m.replace(' ', '_'))) for m in P.MODES]
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Automatch', str(e))
            return
        self._pending = jobs[1:]
        self._all_done = []
        self.start('run', jobs[0])

    # ── running ──────────────────────────────────────────────────────────────
    def start(self, kind: str, preset=None):
        if self.proc is not None:
            return
        if preset is None:
            self._pending, self._all_done = [], []
        try:
            mode, job = preset if preset is not None else (self.mode.currentText(), self.job())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Automatch', str(e))
            return
        self._current = (mode, job['output_dir'])
        os.makedirs(job['output_dir'], exist_ok=True)
        path = os.path.join(job['output_dir'], 'job_preset.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        for key, row in (('input', self.input), ('reference', self.reference), ('reference_c1', self.reference_c1),
                         ('output', self.output), ('truth', self.truth), ('weights', self.weights)):
            self.settings.setValue(key, row.text())
        self.settings.setValue('device', self.device.currentText())
        self.settings.setValue('mode', self.mode.currentText())
        self.settings.setValue('cloud', 'true' if self.cloud.isChecked() else 'false')
        self.settings.setValue('max_err', self.max_err.value())
        self.log.appendPlainText(f'--- {kind}: {self.current_sensor()} preset, {mode}, settings in {path}')
        self._kind, self._buf = kind, ''
        self.bar.setRange(0, 0)
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels if QT_API != 'PyQt5'
                                        else QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read)
        self.proc.finished.connect(self.finished)
        self.btn_run.setEnabled(False)
        self.btn_all.setEnabled(False)
        self.btn_auto.setEnabled(False)
        self.btn_check.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.start(sys.executable, ['-u', JOB_SCRIPT, kind, path])

    def read(self):
        self._buf += bytes(self.proc.readAllStandardOutput()).decode('utf-8', 'replace')
        *lines, self._buf = self._buf.split('\n')
        for line in lines:
            self.line(line.rstrip('\r'))

    def line(self, line: str):
        if not line.startswith(PROGRESS_PREFIX):
            self.log.appendPlainText(line)
            return
        try:
            ev = json.loads(line[len(PROGRESS_PREFIX):])
        except ValueError:
            return
        self.on_event(ev)

    def _bar_for(self, ev):
        """The bar of the GPU an event comes from (two-GPU run), else the main bar."""
        g = ev.get('gpu')
        if g is None or not self._bar_of:
            return self.bar
        if g not in self._bar_of and len(self._bar_of) < len(self.gpu_bars):
            self._bar_of[g] = self.gpu_bars[len(self._bar_of)]
            self._bar_of[g].setVisible(True)
        return self._bar_of.get(g, self.bar)

    def on_event(self, ev: dict):
        kind = ev.get('event')
        gpu = f"GPU {ev['gpu']}: " if ev.get('gpu') is not None else ''
        if kind == 'job':                    # the run's plan: the main bar counts detectors, a bar per GPU
            gpus = list(ev.get('gpus') or [])[:2]
            self._det_done = 0
            self._det_total = len(ev.get('detectors') or []) * max(1, len(ev.get('sweep') or [])) * max(
                1, len(ev.get('channels') or []))
            self._bar_of = {g: b for g, b in zip(gpus, self.gpu_bars)}
            for b in self.gpu_bars:
                b.setVisible(False)
            for g, b in self._bar_of.items():
                b.setVisible(True)
                b.setRange(0, 1)
                b.setValue(0)
                b.setFormat(f'GPU {g}: waiting')
            if self._bar_of:
                self.bar.setRange(0, max(1, self._det_total))
                self.bar.setValue(0)
                self.bar.setFormat(f'detectors 0 / {self._det_total}')
        elif kind == 'detector_done' and self._bar_of:
            self._det_done += 1
            self.bar.setValue(self._det_done)
            self.bar.setFormat(f'detectors {self._det_done} / {self._det_total}')
        if kind in ('window', 'coarse', 'preprocess', 'h5_read') and ev.get('total'):
            bar = self._bar_for(ev)
            bar.setRange(0, int(ev['total']))
            bar.setValue(int(ev.get('done') or 0))
            if kind == 'window':
                where = f" pair {ev.get('pair_no')}/{ev['pairs']}," if ev.get('pairs') else ''
                bar.setFormat(f"{gpu}{ev.get('detector', '')}{where} window %v/%m")
            elif kind == 'coarse':
                bar.setFormat(f"{gpu}coarse offsets: {ev.get('detector', '')}  %v/%m")
            else:
                bar.setFormat(f"{gpu}{kind}  %v/%m")
        if kind == 'detector':
            self.stage.setText(f"{gpu}{ev.get('channel')}: {ev.get('detector')}")
        elif kind == 'stage':
            self.stage.setText(f"{gpu}{ev.get('channel')}: {ev.get('detector')} ({ev.get('stage')})")
        elif kind == 'failed':
            self.stage.setText('not run: ' + '; '.join(ev.get('errors') or []))
        elif kind == 'done':
            self.stage.setText(f"done in {ev.get('minutes')} min")

    def stop(self):
        self._pending = []                       # Stop ends the three-preset sequence too
        self._auto = False
        if self.proc is not None:
            kill_tree(self.proc)                 # the run and its GPU workers

    def results(self, out: str = ''):
        """The files for RIVAL in the output folder."""
        out = out or self.output.text()
        names = sorted(os.listdir(out)) if os.path.isdir(out) else []
        return ([os.path.join(out, n) for n in names if n.startswith('RIVAL_FINAL_') and '_detail' not in n],
                [os.path.join(out, n) for n in names if n.startswith('RIVAL_BEST_FINAL_')])

    def finished(self, code, *_):
        if self._buf:
            self.line(self._buf)
        kind, self.proc = self._kind, None
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if code == 0 else 0)
        self.bar.setFormat('%p%')
        for b in self.gpu_bars:
            b.setVisible(False)
        self._bar_of = {}
        self.btn_all.setEnabled(True)
        self.btn_auto.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_check.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if kind == 'preflight':
            self.stage.setText('check passed: ready to run' if code == 0 else 'check found problems (see the log)')
            return
        mode, out = getattr(self, '_current', ('', ''))
        final, best = self.results(out)
        if getattr(self, '_auto', False) and mode == 'fastest':
            self._auto = False
            if code == 0:
                share = P.rigid_share(out)
                nxt, why = P.auto_next(share)
                self.log.appendPlainText(f'--- auto: {why}')
                try:
                    with open(os.path.join(self.output.text(), 'AUTO_PRESET.txt'), 'w', encoding='utf-8') as f:
                        f.write(f"rigid share: {share}\nchosen: {nxt or 'fastest'}\n{why}\n")
                except OSError:
                    pass
                if nxt:
                    try:
                        self.start('run', (nxt, self.job(nxt, nxt.replace(' ', '_'))))
                        return
                    except ValueError as e:
                        self.log.appendPlainText(f'--- auto: {nxt} not started ({e})')
        if getattr(self, '_pending', None) is not None and (self._pending or self._all_done):
            self._all_done.append((mode, out, code))
            if code == 0 and self._pending:
                nxt, self._pending = self._pending[0], self._pending[1:]
                self.log.appendPlainText(f'--- {mode} done; next: {nxt[0]} (reusing the cache and the detectors '
                                         f'already run)')
                self.start('run', nxt)
                return
            self.log.appendPlainText('--- the presets: ' + '; '.join(
                f"{m}: {'done' if c == 0 else f'ended with code {c}'} -> {o}" for m, o, c in self._all_done)
                + ' (REPORT/REPORT_STATS.csv in each: the points against the manual ones)')
            self._pending, self._all_done = [], []
        if code == 0 and out and kind == 'run':
            self._done_box = show_job_done(self, out)
        if code == 0 and (final or best):
            self.log.appendPlainText('--- for RIVAL:')
            for p in final:
                self.log.appendPlainText(f'    {p}   (points the top detectors agree on)')
            for p in best:
                self.log.appendPlainText(f'    {p}   (the best single detector)')
            self.stage.setText('done: load ' + os.path.basename((final or best)[0]) + ' in RIVAL')
        elif code != 0:
            self.stage.setText(f'ended with code {code} (see the log)')

    def open_output(self):
        if os.path.isdir(self.output.text()):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.output.text()))

    def presets_file(self):
        if not os.path.exists(P.PRESETS_FILE):
            with open(P.PRESETS_FILE, 'w', encoding='utf-8') as f:
                json.dump(P.load(''), f, indent=2)
        QtWidgets.QMessageBox.information(
            self, 'Presets', f'The presets are in\n{P.PRESETS_FILE}\n\nEdit it (per sensor: detectors, window_sizes, '
                             f'channels, ...) and open this window again. Delete it to go back to the built-in ones.')


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = RunWindow()
    w.show()
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == '__main__':
    sys.exit(main())
