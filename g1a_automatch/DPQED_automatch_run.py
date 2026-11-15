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
        self.reference = PathRow(True, 'Reference folder')
        self.output = PathRow(True, 'Output folder')
        self.truth = PathRow(False, 'Ground truth (optional)', 'CSV (*.csv);;All files (*)')
        self.weights = PathRow(True, 'Weights folder (optional)')
        f.addRow('Input image', self.input)
        f.addRow('Reference folder', self.reference)
        f.addRow('Output folder', self.output)
        f.addRow('Ground truth (optional)', self.truth)
        f.addRow('Weights folder (optional)', self.weights)
        self.sensor = QtWidgets.QComboBox()
        self.sensor.addItems(['from the file name'] + list(self.presets))
        self.sensor.setToolTip('The preset used. From the file name: G1A_MV_ / G1A_HV_ / G1A_HS_ / G1A_ML_')
        self.bands = QtWidgets.QLineEdit()
        self.bands.setPlaceholderText("the preset's (or the truth file's)")
        self.bands.setMaximumWidth(220)
        self.bands.setToolTip('Bands to match, e.g. 28 or 28, 50. Empty: the band in the truth file\'s name '
                              '(pts_band28.csv), else the preset\'s')
        self.max_err = QtWidgets.QDoubleSpinBox()
        self.max_err.setRange(1.0, 500.0)
        self.max_err.setValue(25.0)
        self.max_err.setSuffix(' km')
        self.max_err.setToolTip('Largest geolocation error expected. With a ground-truth file it is taken from the '
                                'file instead')
        self.device = QtWidgets.QComboBox()
        self.device.addItems(DEVICES)
        self.device.setToolTip('all: every GPU of this machine, sharing one queue of the detectors')
        self.cloud = QtWidgets.QCheckBox('mask clouds')
        self.cloud.setChecked(True)
        self.cloud.setToolTip('By the sensor: MV OmniCloudMask on bands 5, 4, 6 (needs it installed); HV VNIR, '
                              'HS SWIR, ML infrared tests')
        h = QtWidgets.QHBoxLayout()
        for label, wid in (('Sensor', self.sensor), ('Band(s)', self.bands), ('Max error (no truth)', self.max_err),
                           ('Device', self.device)):
            h.addWidget(QtWidgets.QLabel(label))
            h.addWidget(wid)
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
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_open = QtWidgets.QPushButton('Open output folder')
        self.btn_presets = QtWidgets.QPushButton('Presets file…')
        self.btn_presets.setToolTip('Write automatch_presets.json (the presets, to edit) or show where it is; '
                                    'it is read when this window opens')
        self.btn_stop.setEnabled(False)
        for b in (self.btn_check, self.btn_run, self.btn_stop, self.btn_open, self.btn_presets):
            hb.addWidget(b)
        hb.addStretch(1)
        v.addLayout(hb)
        self.bar = QtWidgets.QProgressBar()
        self.stage = QtWidgets.QLabel('')
        v.addWidget(self.bar)
        v.addWidget(self.stage)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        v.addWidget(self.log, 1)
        self.btn_check.clicked.connect(lambda: self.start('preflight'))
        self.btn_run.clicked.connect(lambda: self.start('run'))
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open.clicked.connect(self.open_output)
        self.btn_presets.clicked.connect(self.presets_file)
        self.input.edit.textChanged.connect(self.refresh)
        self.sensor.currentIndexChanged.connect(self.refresh)
        for key, row in (('input', self.input), ('reference', self.reference), ('output', self.output),
                         ('truth', self.truth), ('weights', self.weights)):
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
            self.summary.setText(f'<b>{s} preset</b><br>' + P.describe(self.presets[s]).replace('\n', '<br>'))
        else:
            self.summary.setText('Sensor not known from the file name (G1A_MV_ / HV / HS / ML): choose it under Sensor.')

    def job(self):
        s = self.current_sensor()
        if s not in self.presets:
            raise ValueError('choose the sensor (the file name does not say it)')
        for label, row in (('input image', self.input), ('reference folder', self.reference),
                           ('output folder', self.output)):
            if not row.text():
                raise ValueError(f'set the {label}')
        job, _ = P.job_for(s, self.input.text(), self.reference.text(), self.output.text(), self.truth.text(),
                           P.parse_bands(self.bands.text()), None if self.truth.text() else self.max_err.value(),
                           self.device.currentText(), self.cloud.isChecked(), self.weights.text(), self.presets)
        return job

    # ── running ──────────────────────────────────────────────────────────────
    def start(self, kind: str):
        if self.proc is not None:
            return
        try:
            job = self.job()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Automatch', str(e))
            return
        os.makedirs(job['output_dir'], exist_ok=True)
        path = os.path.join(job['output_dir'], 'job_preset.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
        for key, row in (('input', self.input), ('reference', self.reference), ('output', self.output),
                         ('truth', self.truth), ('weights', self.weights)):
            self.settings.setValue(key, row.text())
        self.settings.setValue('device', self.device.currentText())
        self.settings.setValue('cloud', 'true' if self.cloud.isChecked() else 'false')
        self.settings.setValue('max_err', self.max_err.value())
        self.log.appendPlainText(f'--- {kind}: {self.current_sensor()} preset, settings in {path}')
        self._kind, self._buf = kind, ''
        self.bar.setRange(0, 0)
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels if QT_API != 'PyQt5'
                                        else QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read)
        self.proc.finished.connect(self.finished)
        self.btn_run.setEnabled(False)
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

    def on_event(self, ev: dict):
        kind = ev.get('event')
        gpu = f"GPU {ev['gpu']}: " if ev.get('gpu') is not None else ''
        if kind in ('window', 'coarse', 'preprocess', 'h5_read') and ev.get('total'):
            self.bar.setRange(0, int(ev['total']))
            self.bar.setValue(int(ev.get('done') or 0))
        if kind == 'detector':
            self.stage.setText(f"{gpu}{ev.get('channel')}: {ev.get('detector')}")
        elif kind == 'stage':
            self.stage.setText(f"{gpu}{ev.get('channel')}: {ev.get('detector')} ({ev.get('stage')})")
        elif kind == 'failed':
            self.stage.setText('not run: ' + '; '.join(ev.get('errors') or []))
        elif kind == 'done':
            self.stage.setText(f"done in {ev.get('minutes')} min")

    def stop(self):
        if self.proc is not None:
            self.proc.kill()

    def results(self):
        """The files for RIVAL in the output folder."""
        out = self.output.text()
        names = sorted(os.listdir(out)) if os.path.isdir(out) else []
        return ([os.path.join(out, n) for n in names if n.startswith('RIVAL_FINAL_') and '_detail' not in n],
                [os.path.join(out, n) for n in names if n.startswith('RIVAL_BEST_FINAL_')])

    def finished(self, code, *_):
        if self._buf:
            self.line(self._buf)
        kind, self.proc = self._kind, None
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if code == 0 else 0)
        self.btn_run.setEnabled(True)
        self.btn_check.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if kind == 'preflight':
            self.stage.setText('check passed: ready to run' if code == 0 else 'check found problems (see the log)')
            return
        final, best = self.results()
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
