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
            m = float(r.get('ce90_m') or 0) < 1000.0          # metres when the error is metres

            def ln(v, m=m):
                return f'{float(v or 0):.0f} m' if m else f'{float(v or 0) / 1000:.3f} km'
            lines.append(f"{r['file']}: mean dE {ln(r.get('mean_dx_m'))}, dN {ln(r.get('mean_dy_m'))}, "
                         f"CE90 {ln(r.get('ce90_m'))}"
                         + (f"  (manual CE90 {ln(man['ce90_m'].iloc[0])})" if len(man) else ''))
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
        self.cache = PathRow(True, 'Cache folder (optional)')
        self.cache.setToolTip('Where references, pairs, the cloud mask and detector results are kept and reused '
                              '(across presets, auto steps and later runs of the same scene). Empty: '
                              '<output>/_cache. Each product gets <cache>/<product name>/ (input folder + file name)')
        f.addRow('Input image', self.input)
        f.addRow('Reference folder (L8)', self.reference)
        f.addRow('C1 references (India)', self.reference_c1)
        self.reference_c1.setToolTip('Optional: with a scene centre in the Indian region (68-98 E, 6-38 N) the job '
                                     'uses these references instead of the L8 ones')
        f.addRow('Output folder', self.output)
        f.addRow('Ground truth (optional)', self.truth)
        f.addRow('Weights folder (optional)', self.weights)
        f.addRow('Cache folder (optional)', self.cache)
        self.sensor = QtWidgets.QComboBox()
        self.sensor.addItems(['from the file name'] + list(self.presets))
        self.sensor.setToolTip('The preset used. From the file name: G1A_MV_ / G1A_HV_ / G1A_HS_ / G1A_ML_')
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(list(P.MODES))
        self.mode.setToolTip('Detectors ranked by truth accuracy per sensor: fastest runs the top 5 (3 agreeing), '
                             'fast the top 10 (3 agreeing), medium the top 20 (4 agreeing), slow all 30 (4 agreeing, '
                             'incl. OmniGlue, DKM, RoMa). The same RANSAC and sub-cells: a tier reuses the detectors '
                             'the tiers below it ran')
        self.mode.setCurrentText(P.MODE_ALIASES.get(self.settings.value('mode', 'fast') or 'fast', self.settings.value('mode', 'fast') or 'fast'))
        self.mode.currentIndexChanged.connect(lambda *_: (setattr(self, '_shown_sensor', None), self.refresh()))
        self.bands = QtWidgets.QLineEdit()
        self.bands.setPlaceholderText('e.g. 28 or 28, 50')
        self.bands.setMaximumWidth(160)
        self.bands.setToolTip('The band(s) to match: yours to name. Empty is accepted only with crispest ticked, '
                              'or with a truth file whose name ends in the band (pts_band28.csv)')
        self.kp_test = QtWidgets.QCheckBox('keypoint test (also 5000 / MP)')
        self.kp_test.setToolTip('Run each detector twice in the same job: at the usual keypoint density and at '
                                '5000 per megapixel (one window size), each in its own sweep folder '
                                '(win<W>_nfauto, win<W>_nf<N>). With a truth file, KEYPOINT_TEST.png in the output '
                                'folder compares their truth error and time per window')
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
        self.cloud_conf = QtWidgets.QDoubleSpinBox()
        self.cloud_conf.setRange(0.3, 0.99)
        self.cloud_conf.setSingleStep(0.05)
        self.cloud_conf.setValue(0.7)
        self.cloud_conf.setPrefix('sure ≥ ')
        self.cloud_conf.setToolTip('Cloud mask threshold: a pixel is cloud / shadow only when the mask is this sure '
                                   '(0.7 default). Raise it (0.85) when bright ground is masked as cloud, lower it '
                                   '(0.5) when thin cloud gets through')
        self.cloud.toggled.connect(self.cloud_conf.setEnabled)
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
        h.addWidget(self.cloud_conf)
        h.addWidget(self.kp_test)
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
        self.btn_all = QtWidgets.QPushButton('Run all 4 tiers')
        self.btn_auto = QtWidgets.QPushButton('Run auto (top 3 → 5 → 10 → 20)')
        self.btn_auto.setToolTip('Top 3 with 2 agreeing; if that fails, top 5 with 3 agreeing; then top 10 with 5; '
                                 'then top 20 with 5. A step passes when the cross-check is reliable and the final '
                                 'points fill 80 % of the sub-tiles. One cache: each step only adds detectors. Each '
                                 'step into <output>/auto_topN_agreeK; the outcome in AUTO_PRESET.txt')
        self.btn_all.setToolTip('fastest, fast, medium, then slow, into <output>/fastest, /fast, /medium, /slow, '
                                'sharing one cache (the cache folder, else <output>/_cache): references, pairs and the cloud '
                                'mask are made once, and a detector one preset ran is reused by the next')
        self.btn_truth = QtWidgets.QPushButton('Compare with truth')
        self.btn_truth.setToolTip('Score the finished run(s) in the output folder (also its auto steps / tiers) '
                                  'against the truth file, without running them again: TRUTH_RANKING, the final '
                                  'points\' truth error, REPORT with the manual points; TRUTH_CHECK.png in the '
                                  'output folder. (Running again with a truth file processes it again: the truth '
                                  'file also sets the window, the search range and the band.)')
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_open = QtWidgets.QPushButton('Open output folder')
        self.btn_presets = QtWidgets.QPushButton('Presets file…')
        self.btn_presets.setToolTip('Write automatch_presets.json (the presets, to edit) or show where it is; '
                                    'it is read when this window opens')
        self.btn_stop.setEnabled(False)
        for b in (self.btn_check, self.btn_run, self.btn_all, self.btn_auto, self.btn_truth, self.btn_stop,
                  self.btn_open, self.btn_presets):
            hb.addWidget(b)
        hb.addStretch(1)
        v.addLayout(hb)
        # the queue: runs one after another, each with the window's settings as they were when added
        self.queue_list = QtWidgets.QListWidget()
        self.queue_list.setMaximumHeight(90)
        self.queue_list.setToolTip('Queued runs, top first: each keeps the settings of this window as they were when '
                                   'added (input, output, band, preset, ...)')
        hq = QtWidgets.QHBoxLayout()
        self.queue_what = QtWidgets.QComboBox()
        self.queue_what.addItems(['Run', 'Run auto', 'Run all 4 tiers'])
        self.btn_q_add = QtWidgets.QPushButton('Add to queue')
        self.btn_q_add.setToolTip('Queue a run of the settings as they are now; then change them for the next one')
        self.btn_q_remove = QtWidgets.QPushButton('Remove')
        self.btn_q_up = QtWidgets.QPushButton('Up')
        self.btn_q_down = QtWidgets.QPushButton('Down')
        self.btn_q_load = QtWidgets.QPushButton('Edit')
        self.btn_q_load.setToolTip('Put the selected run\'s settings in the window to change them (also: double-click '
                                   'it), then Save to queue: it keeps its place in the queue')
        self.btn_q_save = QtWidgets.QPushButton('Save to queue')
        self.btn_q_save.setToolTip('Put the run being edited back in its place with the settings as they are now')
        self.btn_q_cancel = QtWidgets.QPushButton('Cancel edit')
        for b in (self.btn_q_save, self.btn_q_cancel):
            b.setVisible(False)
        self.btn_q_run = QtWidgets.QPushButton('Run the queue')
        self.btn_q_run.setToolTip('Start the queued runs one after another (Stop pauses the queue)')
        for wq in (QtWidgets.QLabel('Queue:'), self.queue_what, self.btn_q_add, self.btn_q_up, self.btn_q_down,
                   self.btn_q_remove, self.btn_q_load, self.btn_q_save, self.btn_q_cancel, self.btn_q_run):
            hq.addWidget(wq)
        hq.addStretch(1)
        v.addLayout(hq)
        v.addWidget(self.queue_list)
        self.btn_q_add.clicked.connect(self.queue_add)
        self.btn_q_remove.clicked.connect(self.queue_remove)
        self.btn_q_up.clicked.connect(self.queue_up)
        self.btn_q_down.clicked.connect(self.queue_down)
        self.btn_q_load.clicked.connect(lambda: self.queue_edit(self.queue_list.currentRow()))
        self.queue_list.itemDoubleClicked.connect(lambda *_: self.queue_edit(self.queue_list.currentRow()))
        self.btn_q_save.clicked.connect(self.queue_save)
        self.btn_q_cancel.clicked.connect(self.queue_cancel_edit)
        self._q_edit = None                    # the queued entry being edited
        self.btn_q_run.clicked.connect(self.queue_start)
        self._wq, self._q_running = [], False
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
        if P.LOAD_NOTE:
            self.log.appendPlainText('--- presets: ' + P.LOAD_NOTE)
        self.btn_check.clicked.connect(lambda: self.start('preflight'))
        self.btn_run.clicked.connect(lambda: self.start('run'))
        self.btn_all.clicked.connect(self.run_all)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_truth.clicked.connect(self.compare_truth)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open.clicked.connect(self.open_output)
        self.btn_presets.clicked.connect(self.presets_file)
        self.input.edit.textChanged.connect(self.refresh)
        self.sensor.currentIndexChanged.connect(self.refresh)
        for key, row in (('input', self.input), ('reference', self.reference), ('reference_c1', self.reference_c1),
                         ('output', self.output), ('truth', self.truth), ('weights', self.weights),
                         ('cache', self.cache)):
            row.edit.setText(self.settings.value(key, '') or '')
        self.device.setCurrentText(self.settings.value('device', 'all') or 'all')
        self.cloud.setChecked(str(self.settings.value('cloud', 'true')).lower() == 'true')
        try:
            self.cloud_conf.setValue(float(self.settings.value('cloud_conf', 0.7)))
            self._wq = json.loads(self.settings.value('run_queue', '[]') or '[]')
        except (TypeError, ValueError):
            pass
        self._queue_show()
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
            if s != getattr(self, '_shown_sensor', None) or m != getattr(self, '_shown_mode', None):     # the preset's own numbers, until changed here
                self.top_n.setValue(int(p.get('top_n', 5) or 5))
                self.agree.setValue(int(p.get('agree', 3) or 3))
                self._shown_sensor, self._shown_mode = s, m
            self.summary.setText(f'<b>{s} preset, {m}</b> ({P.BASIS})<br>' + P.describe(p).replace('\n', '<br>'))
        else:
            self.summary.setText('Sensor not known from the file name (G1A_MV_ / HV / HS / ML): choose it under Sensor.')

    def job(self, mode: str = '', sub: str = '', top_agree=None):
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
                           top_n=(top_agree[0] if top_agree else None if mode else self.top_n.value()),
                           agree=(top_agree[1] if top_agree else None if mode else self.agree.value()),
                           mode=mode or self.mode.currentText(), final_points=self.final_points.value(),
                           cell_px=P.parse_levels(self.cell_px.text()),
                           cloud_conf=self.cloud_conf.value() if self.cloud.isChecked() else None,
                           keypoint_test=5000 if self.kp_test.isChecked() else 0,
                           temp_dir=self.cache_dir())
        if self.cache.text().strip():                 # <cache>/<product name>/
            job['temp_per_product'] = True
        if '[window' in note:
            self.log.appendPlainText('--- ' + note[note.index('[window') + 1:].rstrip(']'))
        return job

    def auto_job(self, i: int):
        n, k = P.AUTO_STEPS[i]
        return P.auto_step_name(i), self.job('slow', P.auto_step_name(i), (n, k))

    def run_auto(self):
        """The auto ladder (P.AUTO_STEPS): top 3 / 2 agreeing, then top 5 / 3, top 10 / 5, top 20 / 5,
        each only when the one before failed (P.step_passes)."""
        if self.proc is not None:
            return
        try:
            step = self.auto_job(0)
        except ValueError as e:
            self.warn(str(e))
            return
        self._pending, self._all_done = [], []
        self._auto = 0
        self._auto_log = []
        self.start('run', step)

    def run_all(self):
        """The three presets one after the other (their own top X / Y), each
        into its own subfolder, one cache: the later ones reuse what the
        earlier ones made."""
        if self.proc is not None:
            return
        try:
            jobs = [(m, self.job(m, m.replace(' ', '_'))) for m in P.MODES]
        except ValueError as e:
            self.warn(str(e))
            return
        self._pending = jobs[1:]
        self._all_done = []
        self._auto = None
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
            self.warn(str(e))
            return
        self._current = (mode, job['output_dir'])
        os.makedirs(job['output_dir'], exist_ok=True)
        path = os.path.join(job['output_dir'], 'job_preset.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        for key, row in (('input', self.input), ('reference', self.reference), ('reference_c1', self.reference_c1),
                         ('output', self.output), ('truth', self.truth), ('weights', self.weights),
                         ('cache', self.cache)):
            self.settings.setValue(key, row.text())
        self.settings.setValue('device', self.device.currentText())
        self.settings.setValue('mode', self.mode.currentText())
        self.settings.setValue('cloud', 'true' if self.cloud.isChecked() else 'false')
        self.settings.setValue('cloud_conf', self.cloud_conf.value())
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

    def compare_truth(self):
        """The finished run(s) in the output folder against the truth file (automatch_truthcheck)."""
        if self.proc is not None:
            return
        out, truth = self.output.text().strip(), self.truth.text().strip()
        if not (out and os.path.isdir(out) and truth and os.path.isfile(truth)):
            QtWidgets.QMessageBox.warning(self, 'Compare with truth', 'Set the output folder of the finished run(s) '
                                                                      'and the ground truth file.')
            return
        self.log.appendPlainText(f'--- compare with truth: {out} against {truth} (nothing is run again)')
        self._kind, self._buf, self._current = 'truthcheck', '', ('', out)
        self.bar.setRange(0, 0)
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels if QT_API != 'PyQt5'
                                        else QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read)
        self.proc.finished.connect(self.finished)
        for b in (self.btn_run, self.btn_all, self.btn_auto, self.btn_check, self.btn_truth):
            b.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.start(sys.executable, ['-u', JOB_SCRIPT, 'truthcheck', out, '--truth', truth])

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
        self._pending = []                       # Stop ends the preset sequence too
        self._auto = None
        if self._q_running:
            self._q_running = False
            self.log.appendPlainText(f'--- queue paused ({len(self._wq)} queued): Run the queue to go on')
        if self.proc is not None:
            kill_tree(self.proc)                 # the run and its GPU workers

    def results(self, out: str = ''):
        """The files for RIVAL in the output folder."""
        out = out or self.output.text()
        names = sorted(os.listdir(out)) if os.path.isdir(out) else []
        try:
            import automatch_cross as C
            prod = C.product_name(out)
        except Exception:
            prod = ''
        named = [os.path.join(out, n) for n in names if prod and n.startswith(prod + '_') and n.endswith('.csv')]
        return (sorted(named, key=len) + [os.path.join(out, n) for n in names
                                          if n.startswith('RIVAL_FINAL_') and '_detail' not in n],
                [os.path.join(out, n) for n in names if n.startswith('RIVAL_BEST_FINAL_')])

    def finished(self, code, *_):
        self._finished(code)
        if self.proc is None and self._q_running:       # the run (or its sequence) is over: the next queued
            QtCore.QTimer.singleShot(0, self.queue_next)

    def _finished(self, code):
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
        self.btn_truth.setEnabled(True)
        if kind == 'truthcheck':
            png = os.path.join(getattr(self, '_current', ('', ''))[1] or self.output.text(), 'TRUTH_CHECK.png')
            self.stage.setText('compared with the truth: TRUTH_CHECK.png and each run\'s TRUTH_RANKING.csv, REPORT/'
                               if code == 0 else 'compare with truth: not done (see the log)')
            if code == 0 and os.path.exists(png):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(png))
            return
        if kind == 'preflight':
            self.stage.setText('check passed: ready to run' if code == 0 else 'check found problems (see the log)')
            return
        mode, out = getattr(self, '_current', ('', ''))
        final, best = self.results(out)
        i = getattr(self, '_auto', None)
        if i is not None and kind == 'run':
            ok, why = P.step_passes(out) if code == 0 else (False, f'the run failed (exit {code})')
            self._auto_log.append(f'{mode}: {"passed" if ok else "failed"}: {why}')
            self.log.appendPlainText(f'--- auto: {mode}: {"passed" if ok else "failed"}: {why}')
            last = ok or i + 1 >= len(P.AUTO_STEPS) or code != 0
            if last:
                self._auto = None
                chosen = mode if ok else f'none passed; the last step ({mode}) is the result'
                try:
                    with open(os.path.join(self.output.text(), 'AUTO_PRESET.txt'), 'w', encoding='utf-8') as f:
                        f.write(f"chosen: {chosen}\nrigid share: {P.rigid_share(out)}\n" + '\n'.join(self._auto_log) + '\n')
                except OSError:
                    pass
            else:
                self._auto = i + 1
                try:
                    self.start('run', self.auto_job(i + 1))
                    return
                except ValueError as e:
                    self._auto = None
                    self.log.appendPlainText(f'--- auto: next step not started ({e})')
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

    def cache_dir(self) -> str:
        """The cache folder: the one chosen, else <output>/_cache."""
        return self.cache.text().strip() or os.path.join(self.output.text(), '_cache')

    # ── the queue ────────────────────────────────────────────────────────────
    def fields(self) -> dict:
        """The window's settings, to queue and restore."""
        return {'input': self.input.text(), 'reference': self.reference.text(),
                'reference_c1': self.reference_c1.text(), 'output': self.output.text(), 'truth': self.truth.text(),
                'weights': self.weights.text(), 'cache': self.cache.text(),
                'sensor': self.sensor.currentText(), 'mode': self.mode.currentText(),
                'bands': self.bands.text(), 'crispest': self.crispest.isChecked(), 'max_err': self.max_err.value(),
                'device': self.device.currentText(), 'top_n': self.top_n.value(), 'agree': self.agree.value(),
                'final_points': self.final_points.value(), 'cell_px': self.cell_px.text(),
                'cloud': self.cloud.isChecked(), 'cloud_conf': self.cloud_conf.value(),
                'kp_test': self.kp_test.isChecked()}

    def set_fields(self, f: dict):
        for key, row in (('input', self.input), ('reference', self.reference), ('reference_c1', self.reference_c1),
                         ('output', self.output), ('truth', self.truth), ('weights', self.weights),
                         ('cache', self.cache)):
            row.edit.setText(f.get(key, ''))
        self.sensor.setCurrentText(f.get('sensor', self.sensor.currentText()))
        self.mode.setCurrentText(f.get('mode', self.mode.currentText()))
        self.bands.setText(f.get('bands', ''))
        self.crispest.setChecked(bool(f.get('crispest')))
        self.max_err.setValue(float(f.get('max_err', self.max_err.value())))
        self.device.setCurrentText(f.get('device', self.device.currentText()))
        self.top_n.setValue(int(f.get('top_n', self.top_n.value())))       # after the preset: its numbers reset these
        self.agree.setValue(int(f.get('agree', self.agree.value())))
        self.final_points.setValue(int(f.get('final_points', self.final_points.value())))
        self.cell_px.setText(f.get('cell_px', self.cell_px.text()))
        self.cloud.setChecked(bool(f.get('cloud', True)))
        self.cloud_conf.setValue(float(f.get('cloud_conf', 0.7)))
        self.kp_test.setChecked(bool(f.get('kp_test', False)))

    def _queue_show(self):
        self.queue_list.clear()
        for e in self._wq:
            f = e['fields']
            self.queue_list.addItem(f"{e['what']}: {os.path.basename(f.get('input', ''))}  band {f.get('bands') or '-'}"
                                    f"  {f.get('mode', '')}  -> {f.get('output', '')}")
        self.settings.setValue('run_queue', json.dumps(self._wq))

    def warn(self, msg: str):
        """A run that cannot start: while the queue runs, logged (the queue goes on
        unattended); otherwise a message."""
        if getattr(self, '_q_running', False):
            self.log.appendPlainText(f'--- queue: not started: {msg}')
        else:
            QtWidgets.QMessageBox.warning(self, 'Automatch', msg)

    def queue_problem(self, what: str) -> str:
        """Why the window's settings cannot run as queued ('' when they can): the
        same check a run makes when it starts (band, folders, sensor)."""
        try:
            if what == 'Run auto':
                self.auto_job(0)
            elif what == 'Run all 4 tiers':
                self.job('fastest', 'fastest')
            else:
                self.job()
        except ValueError as e:
            return str(e)
        return ''

    def _queue_entry(self, skip=None):
        """The window's settings as a queue entry, or None (a message said why)."""
        f = self.fields()
        if not (f['input'] and f['output']):
            QtWidgets.QMessageBox.warning(self, 'Queue', 'set the input image and the output folder first')
            return None
        same = [e for e in self._wq if e is not skip and os.path.normcase(os.path.normpath(e['fields']['output'])) ==
                os.path.normcase(os.path.normpath(f['output']))]
        if same:
            QtWidgets.QMessageBox.warning(self, 'Queue', f"{f['output']} is already the output of a queued run: "
                                                         'choose another output folder')
            return None
        why = self.queue_problem(self.queue_what.currentText())
        if why:                                  # caught now, not when the queue reaches it unattended
            QtWidgets.QMessageBox.warning(self, 'Queue', f'This run could not start: {why}')
            return None
        return {'what': self.queue_what.currentText(), 'fields': f}

    def queue_add(self):
        e = self._queue_entry()
        if e is not None:
            self._wq.append(e)
            self._queue_show()

    def queue_edit(self, i: int):
        """The selected run's settings into the window, to change and save back in its place."""
        if not 0 <= i < len(self._wq):
            return
        self._q_edit = self._wq[i]
        self.set_fields(self._q_edit['fields'])
        self.queue_what.setCurrentText(self._q_edit['what'])
        for b in (self.btn_q_save, self.btn_q_cancel):
            b.setVisible(True)
        self.log.appendPlainText(f'--- queue: editing run {i + 1} ({os.path.basename(self._q_edit["fields"].get("input", ""))}); '
                                 f'change the settings, then Save to queue')

    def queue_save(self):
        if self._q_edit is None:
            return
        if not any(e is self._q_edit for e in self._wq):
            QtWidgets.QMessageBox.information(self, 'Queue', 'That run has already started (or was removed): '
                                                             'Add to queue to queue it again.')
            self.queue_cancel_edit()
            return
        e = self._queue_entry(skip=self._q_edit)
        if e is None:
            return
        i = next(k for k, x in enumerate(self._wq) if x is self._q_edit)
        self._wq[i] = e
        self._queue_show()
        self.queue_list.setCurrentRow(i)
        self.queue_cancel_edit()

    def queue_cancel_edit(self):
        self._q_edit = None
        for b in (self.btn_q_save, self.btn_q_cancel):
            b.setVisible(False)

    def queue_down(self):
        i = self.queue_list.currentRow()
        if 0 <= i < len(self._wq) - 1:
            self._wq[i + 1], self._wq[i] = self._wq[i], self._wq[i + 1]
            self._queue_show()
            self.queue_list.setCurrentRow(i + 1)

    def queue_remove(self):
        i = self.queue_list.currentRow()
        if 0 <= i < len(self._wq):
            del self._wq[i]
            self._queue_show()

    def queue_up(self):
        i = self.queue_list.currentRow()
        if 0 < i < len(self._wq):
            self._wq[i - 1], self._wq[i] = self._wq[i], self._wq[i - 1]
            self._queue_show()
            self.queue_list.setCurrentRow(i - 1)

    def queue_load(self, i: int):
        """The selected run's settings into the window (see queue_edit to change it in place)."""
        if 0 <= i < len(self._wq):
            self.set_fields(self._wq[i]['fields'])

    def queue_start(self):
        self._q_running = True
        self.queue_next()

    def queue_next(self):
        """Start the first queued run when nothing runs (the queue on)."""
        if not self._q_running or self.proc is not None:
            return
        if not self._wq:
            self._q_running = False
            skipped = getattr(self, '_q_skipped', [])
            self.log.appendPlainText('--- queue: all runs done'
                                     + (f'; {len(skipped)} did not start: ' + ', '.join(os.path.basename(x) for x in skipped)
                                        if skipped else ''))
            self._q_skipped = []
            return
        e = self._wq.pop(0)
        if e is self._q_edit:                 # it starts now: the edit ends (with the settings saved)
            self.queue_cancel_edit()
        self._queue_show()
        self.set_fields(e['fields'])
        self.log.appendPlainText(f"--- queue: {e['what']} of {e['fields'].get('input')} "
                                 f"({len(self._wq)} more queued)")
        {'Run auto': self.run_auto, 'Run all 4 tiers': self.run_all}.get(e['what'], lambda: self.start('run'))()
        if self.proc is None:                 # it did not start (a setting missing): on to the next
            self.log.appendPlainText('--- queue: that run did not start (see above); next')
            self._q_skipped = getattr(self, '_q_skipped', []) + [e['fields'].get('input', '')]
            QtCore.QTimer.singleShot(0, self.queue_next)

    def open_output(self):
        if os.path.isdir(self.output.text()):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.output.text()))

    def presets_file(self):
        if not os.path.exists(P.PRESETS_FILE) or P.LOAD_NOTE:     # new, or from an earlier ranking: rewritten
            with open(P.PRESETS_FILE, 'w', encoding='utf-8') as f:
                json.dump({'_version': P.PRESETS_VERSION, **self.presets}, f, indent=2)
            P.LOAD_NOTE = ''
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
