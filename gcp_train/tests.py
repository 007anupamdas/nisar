"""
Checks for gcp_train on synthetic scenes (no Glue Factory needed):

    python tests.py

A reference raster holds the true ground; an input image shows that ground
through a known mapping (its pixel (c, r) images true point T(c, r)) while its
own georeference claims something else, as a misgeolocated satellite image
does. Tie points in the RIVAL layout (space-separated, no header, with unset
rows) are written from T. The fit must recover T, the pairs' H must align the
two crops, and badly marked scenes must be refused.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

FAILS = []


def check(name, got, want=True):
    ok = got == want
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  (got {got!r}, want {want!r})'))
    if not ok:
        FAILS.append(name)


CRS = 'EPSG:32643'


def make_reference(path, rng, size=1600, px=30.0, x0=495000.0, y1=1705000.0):
    import cv2
    import rasterio as rio
    from rasterio.transform import from_origin
    a = np.zeros((size, size), np.float32)
    for s, w in ((1.5, 40), (4, 60), (12, 90), (40, 120)):
        t = cv2.GaussianBlur(rng.normal(size=(size, size)).astype(np.float32), (0, 0), s)
        a += w * t / t.std()
    a = np.clip(1500 + a, 1, 65000).astype(np.uint16)
    with rio.open(path, 'w', driver='GTiff', height=size, width=size, count=1, dtype='uint16', crs=CRS,
                  transform=from_origin(x0, y1, px, px), nodata=0) as d:
        d.write(a, 1)
    return path


def true_map(c, r, quad=False):
    """Input pixel -> true ground (E, N): a misplaced, slightly rotated and scaled image."""
    E = 498000.0 + 60.3 * c + 1.2 * r
    N = 1702000.0 - 0.9 * c - 59.8 * r
    if quad:                                     # error growing across the scene (job 347)
        E = E + 0.004 * c * c
        N = N - 0.003 * c * r
    return E, N


def _bilinear(img, x, y):
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    h, w = img.shape
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, 0)


def make_input_image(path, ref_path, size=600, quad=False, claimed=(500000.0, 1700000.0, 60.0)):
    import rasterio as rio
    from rasterio.transform import from_origin
    with rio.open(ref_path) as r:
        ref = r.read(1).astype(np.float32)
        inv = ~r.transform
    cc, rr = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    E, N = true_map(cc, rr, quad)
    x, y = inv * (E, N)
    a = _bilinear(ref, np.asarray(x) - 0.5, np.asarray(y) - 0.5)
    a = np.clip(a, 0, 65000).astype(np.uint16)
    with rio.open(path, 'w', driver='GTiff', height=size, width=size, count=1, dtype='uint16', crs=CRS,
                  transform=from_origin(claimed[0], claimed[1], claimed[2], claimed[2]), nodata=0) as d:
        d.write(a, 1)
    return path


def write_gt(path, input_path, rng, n=12, quad=False, noise_px=0.0, outlier=None, broken_rows=True):
    import rasterio as rio
    from pyproj import Transformer
    to_ll = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
    with rio.open(input_path) as s:
        c = rng.uniform(40, s.width - 40, n)
        r = rng.uniform(40, s.height - 40, n)
        ix, iy = s.transform * (c + 0.5, r + 0.5)
    E, N = true_map(c, r, quad)
    E = E + rng.normal(0, noise_px * 60.0, n)
    N = N + rng.normal(0, noise_px * 60.0, n)
    if outlier is not None:
        E[outlier] += 5 * 60.0
    ilo, ila = to_ll.transform(np.asarray(ix), np.asarray(iy))
    rlo, rla = to_ll.transform(E, N)
    lines = []
    for k in range(n):
        lines.append(' '.join([f'{ix[k]:.3f}', f'{iy[k]:.3f}', f'{E[k]:.3f}', f'{N[k]:.3f}',
                               f'{ix[k] - E[k]:.3f}', f'{iy[k] - N[k]:.3f}', str(k + 1),
                               f'{ilo[k]:.8f}', f'{ila[k]:.8f}', f'{rlo[k]:.8f}', f'{rla[k]:.8f}']))
    if broken_rows:
        lines[2] = '\t'.join(lines[2].split()[:9])                          # Ref_Lon/Ref_Lat absent, tab-separated
        p = lines[3].split()
        lines[3] = ' '.join(p[:2] + ['0.000', '0.000'] + p[4:9] + ['0', '0'])  # unset reference point
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main():
    import cv2
    import pandas as pd
    import gt as G
    import fit as F
    import pairs as P
    import dataset as D
    import export as X
    import evaluate as V
    import gpuaas as Q
    tmp = tempfile.mkdtemp(prefix='gcp_train_test_')
    rng = np.random.default_rng(7)
    try:
        ref = make_reference(os.path.join(tmp, 'ref.tif'), rng)
        inp = make_input_image(os.path.join(tmp, 'C3_scene1.img.tif'), ref)
        g1 = write_gt(os.path.join(tmp, 'C3_scene1.txt'), inp, rng, outlier=6)
        pts = G.load_points(g1, quiet=True)
        check('gt: whitespace rows read; the tab row missing Ref_Lon/Lat is recovered from its X/Y, '
              'the unset point left out', (len(pts), 3 in pts['point_id'].tolist(), 4 in pts['point_id'].tolist()),
              (11, True, False))
        res = F.fit_scene(pts, inp)
        check('fit: the marked outlier is dropped, the affine kept, leave-one-out under 0.05 px',
              (res['dropped'], res['type'], res['loo_rmse_px'] < 0.05, res['ok']), ([7], 'affine', True, True))
        col, row = np.array([100.0, 450.0]), np.array([80.0, 500.0])
        E, N = res['model'](col, row)
        Et, Nt = true_map(col, row)
        check('fit: the model gives the true ground within 1 m', float(np.hypot(E - Et, N - Nt).max()) < 1.0, True)
        # a scene whose error grows across it needs the quadratic
        inq = make_input_image(os.path.join(tmp, 'C2E_scene2.tif'), ref, quad=True)
        gq = write_gt(os.path.join(tmp, 'C2E_scene2.txt'), inq, rng, n=14, quad=True, broken_rows=False)
        rq = F.fit_scene(G.load_points(gq, quiet=True), inq)
        check('fit: an error growing across the scene is fitted by the quadratic', (rq['type'], rq['ok']),
              ('quadratic', True))
        # badly marked points are refused
        inb = make_input_image(os.path.join(tmp, 'RS2_scene3.tif'), ref)
        gb = write_gt(os.path.join(tmp, 'RS2_scene3.txt'), inb, rng, noise_px=4.0, broken_rows=False)
        rb = F.fit_scene(G.load_points(gb, quiet=True), inb)
        check('fit: points marked 4 px apart are refused (leave-one-out over 1.5 px)',
              (rb['ok'], 'leave-one-out' in rb['why']), (False, True))
        # modality and pixel-size ranges
        import modality as Mo
        check('modality: names, aliases, wavelengths (nm or um) and SAR sensors by name',
              [Mo.resolve('thermal'), Mo.resolve('Optical'), Mo.resolve(None, 10800), Mo.resolve(None, 10.8),
               Mo.resolve(None, 485), Mo.resolve(None, 865), Mo.resolve(None, 1650), Mo.resolve(None, 3900),
               Mo.resolve(None, None, 'RISAT-1A'), Mo.resolve(None, None, 'NISAR_L'), Mo.resolve(None, None, 'MicroSat'),
               Mo.resolve(float('nan'), float('nan'), 'C2E')],
              ['lwir', 'visible', 'lwir', 'lwir', 'visible', 'nir', 'swir', 'mwir', 'sar', 'sar', 'visible', 'visible'])
        try:
            Mo.resolve('infrared-ish')
            bad = False
        except ValueError:
            bad = True
        check('modality: an unknown name is refused, not guessed', bad, True)
        check('pairs: pixel-size limits from a number, a range, a list; empty or 0 means automatic',
              [P.res_spec('45'), P.res_spec('45-315'), P.res_spec([315, 45]), P.res_spec(120.0), P.res_spec(''),
               P.res_spec(0), P.res_spec('auto'), P.res_spec(float('nan')), P.res_spec(None)],
              [(45.0, 45.0), (45.0, 315.0), (45.0, 315.0), (120.0, 120.0), None, None, None, None, None])
        # pairs
        man = pd.DataFrame([
            {'scene_id': 'C3_scene1', 'sensor': 'C3', 'date': '2026-01-01', 'input_path': inp, 'input_band': 1,
             'gt_path': g1, 'reference': '', 'reference_band': 1, 'split': 'train'},
            {'scene_id': 'C2E_scene2', 'sensor': 'C2E', 'date': '2026-01-02', 'input_path': inq, 'input_band': 1,
             'gt_path': gq, 'reference': ref, 'reference_band': 1, 'split': 'test'},
            {'scene_id': 'RS2_scene3', 'sensor': 'RS2', 'date': '2026-01-03', 'input_path': inb, 'input_band': 1,
             'gt_path': gb, 'reference': ref, 'reference_band': 1}])
        mp = os.path.join(tmp, 'scenes.csv')
        refdir = os.path.join(tmp, 'L8_ref')                 # a reference folder, as for a collection
        os.makedirs(refdir)
        shutil.copyfile(ref, os.path.join(refdir, 'ref.tif'))
        man.loc[0, 'reference'] = refdir
        man.to_csv(mp, index=False)
        out = os.path.join(tmp, 'pairs')
        r = P.build(mp, out, crop=192, per_scene=6, shift_frac=0.25, seed=1)
        fitted = pd.read_csv(os.path.join(out, 'scenes_fit.csv')).set_index('scene_id')
        check('pairs: good scenes give pairs, the badly marked one none (and why)',
              (int(fitted.loc['C3_scene1', 'pairs']), int(fitted.loc['C2E_scene2', 'pairs']),
               bool(fitted.loc['RS2_scene3', 'used']), 'leave-one-out' in str(fitted.loc['RS2_scene3', 'why'])),
              (6, 6, False, True))
        idx = pd.read_csv(os.path.join(out, 'pairs.csv'))
        # H aligns the crops: warp crop 1 into crop 0 and correlate
        cors = []
        for rr in idx.to_dict('records'):
            a0 = cv2.imread(os.path.join(out, rr['img0']), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            a1 = cv2.imread(os.path.join(out, rr['img1']), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            H = np.array([float(v) for v in rr['H'].split()]).reshape(3, 3)
            w1 = cv2.warpPerspective(a1, H, a0.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            ok = (w1 > 0) & (a0 > 0)
            x, y = a0[ok] - a0[ok].mean(), w1[ok] - w1[ok].mean()
            cors.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())))
            # and the crops are misaligned before warping (the matcher has work to do)
        check('pairs: H aligns every crop pair (correlation after warping > 0.9)', min(cors) > 0.9, True)
        sh = np.hypot(idx['shift_x_px'], idx['shift_y_px'])
        check('pairs: the reference crops are shifted at random (up to a quarter crop)',
              (float(sh.max()) > 5, float(sh.max()) <= 0.25 * 192 * 1.5), (True, True))
        check('pairs: the split follows the manifest, by scene',
              sorted(idx.groupby('scene_id')['split'].first().to_dict().items()),
              [('C2E_scene2', 'test'), ('C3_scene1', 'train')])
        check('pairs: splits.csv lists every scene', len(pd.read_csv(os.path.join(out, 'splits.csv'))), 3)
        check('pairs: assigned splits are deterministic', P.assign_split('X1') == P.assign_split('X1'), True)
        # pixel size without a setting: from the coarser image (here the 60 m input, the reference is 30 m)
        # up to what lets a crop fit in the points' area (grown by 15 %, inside the image)
        c3 = idx[idx['scene_id'] == 'C3_scene1']
        f1 = fitted.loc['C3_scene1']
        lo, hi = [float(v) for v in str(f1['res_m']).split('-')]
        bc0, br0, bc1, br1 = res['bounds_px']
        side = min(min(600, bc1 + 0.15 * (bc1 - bc0)) - max(0, bc0 - 0.15 * (bc1 - bc0)),
                   min(600, br1 + 0.15 * (br1 - br0)) - max(0, br0 - 0.15 * (br1 - br0)))
        check('pairs: automatic pixel size: floor is the coarser image (input 60 m, reference 30 m), top where a '
              'crop still fits', (round(float(f1['input_px_m'])), round(float(f1['reference_px_m'])), round(lo),
                                  abs(hi - float(f1['input_px_m']) * side / 192) < 0.5), (60, 30, 60, True))
        check('pairs: every crop draws its own pixel size within that range',
              (bool(c3['px_m'].between(lo - 0.01, hi + 0.01).all()), c3['px_m'].nunique() == len(c3),
               float(c3['px_m'].max() / c3['px_m'].min()) > 1.2), (True, True, True))
        # a reference coarser than the image sets the floor: never upsampled into fake detail
        import geo
        import rasterio as rio
        from rasterio.transform import from_origin
        with rio.open(ref) as rs:
            a30 = rs.read(1).astype(np.float32)
        a120 = a30.reshape(400, 4, 400, 4).mean(axis=(1, 3))
        g120, v120 = geo.reproject_to_grid([(ref, 1)], CRS, 495000.0 + 120 * 50, 1705000.0 - 120 * 60, 120.0, 200)
        bl = a30[240 * 1 + 1:240 + 800:4, 201:201 + 800:4][:200, :200]
        check('geo: a reference resampled 4x coarser is averaged (equals the 4 x 4 block mean), not sampled',
              (float(np.abs(g120 - a120[60:260, 50:250]).max()) < 1.0,
               float(np.abs(bl - a120[60:260, 50:250]).mean()) > 5.0), (True, True))
        ref120 = os.path.join(tmp, 'ref120.tif')
        with rio.open(ref120, 'w', driver='GTiff', height=400, width=400, count=1, dtype='float32', crs=CRS,
                      transform=from_origin(495000.0, 1705000.0, 120.0, 120.0), nodata=0) as d:
            d.write(a120, 1)
        man3 = pd.DataFrame([
            {'scene_id': 'C3_scene1', 'sensor': 'C3', 'input_path': inp, 'input_band': 1, 'gt_path': g1,
             'reference': ref120, 'reference_band': 1, 'split': 'train'},
            {'scene_id': 'C3_scene1', 'sensor': 'C3', 'input_path': inp, 'input_band': 1, 'gt_path': g1,
             'reference': ref, 'reference_band': 1, 'split': 'test', 'modality': 'thermal'}])
        mp3 = os.path.join(tmp, 'scenes3.csv')
        man3.to_csv(mp3, index=False)
        out3 = os.path.join(tmp, 'pairs_rows')
        P.build(mp3, out3, crop=96, per_scene=4, seed=3)
        i3 = pd.read_csv(os.path.join(out3, 'pairs.csv'))
        coarse = i3[i3['img0'].str.contains('/b1_0')]
        check('pairs: a 120 m reference makes 120 m the floor', (len(coarse) > 0, float(coarse['px_m'].min()) >= 119.9),
              (True, True))
        check('pairs: two rows of one scene: own file names, the same split, each its own modality',
              (sorted(i3['img0'].str.split('/').str[1].str.split('_').str[:2].str.join('_').unique().tolist()),
               sorted(i3['split'].unique().tolist()), sorted(i3['modality0'].unique().tolist()),
               i3['modality1'].unique().tolist()),
              (['b1_0000', 'b1_0001', 'b1_0002', 'b1_0003', 'b1_2'], ['train'], ['lwir', 'visible'], ['visible']))
        # a working resolution coarser than the image: crops read at 120 m, H still aligns them
        out2 = os.path.join(tmp, 'pairs_120m')
        P.build(mp, out2, crop=96, per_scene=4, seed=2, res_m=120.0)
        i2 = pd.read_csv(os.path.join(out2, 'pairs.csv'))
        f2 = pd.read_csv(os.path.join(out2, 'scenes_fit.csv')).set_index('scene_id')
        cor2 = []
        for rr in i2.to_dict('records'):
            a0 = cv2.imread(os.path.join(out2, rr['img0']), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            a1 = cv2.imread(os.path.join(out2, rr['img1']), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            H = np.array([float(v) for v in rr['H'].split()]).reshape(3, 3)
            w1 = cv2.warpPerspective(a1, H, a0.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            ok = (w1 > 0) & (a0 > 0)
            x, y = a0[ok] - a0[ok].mean(), w1[ok] - w1[ok].mean()
            cor2.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())))
        check('pairs: a fixed pixel size (res_m 120): every crop is 120 m and H still aligns them',
              (float(f2.loc['C3_scene1', 'res_m']), sorted(set(i2['px_m'].round(3))), len(i2) > 0, min(cor2) > 0.9),
              (120.0, [120.0], True, True))
        out4 = os.path.join(tmp, 'pairs_45m')
        P.build(mp, out4, crop=96, per_scene=2, seed=4, res_m=45.0)
        i4 = pd.read_csv(os.path.join(out4, 'pairs.csv'))
        check('pairs: a pixel size finer than the coarser image (45 m for a 60 m image) is raised to it',
              sorted(set(i4[i4['scene_id'] == 'C3_scene1']['px_m'].round())), [60.0])
        # dataset
        ds = D.PairDataset(os.path.join(out, 'pairs.csv'), 'train', channels=3)
        s = ds[0]
        check('dataset: Glue Factory sample layout (3-channel views, image_size, H)',
              (tuple(s['view0']['image'].shape), s['view0']['image_size'].tolist(), tuple(s['H_0to1'].shape)),
              ((3, 192, 192), [192.0, 192.0], (3, 3)))
        dt = D.PairDataset(os.path.join(out, 'pairs.csv'), 'test', channels=1)
        check('dataset: the test split, unaugmented, one channel', (len(dt), tuple(dt[0]['view1']['image'].shape)),
              (6, (1, 192, 192)))
        d3 = D.PairDataset(os.path.join(out3, 'pairs.csv'), 'train', channels=3, modalities=['thermal'])
        seen = set()
        for k in range(40):
            x = d3[k % len(d3)]
            seen.add((x['modality0'], x['modality1']))
        check('dataset: each sample carries both modalities and the pixel size; a swap exchanges the modalities; '
              'modalities filter the inputs', (len(d3), sorted(seen), isinstance(x['px_m'], float)),
              (4, [(0, 4), (4, 0)], True))
        # metrics
        k0 = rng.uniform(0, 191, (50, 2))
        Ht = np.array([float(v) for v in idx['H'].iloc[0].split()]).reshape(3, 3)
        q = np.column_stack([k0, np.ones(50)]) @ Ht.T
        k1 = q[:, :2] / q[:, 2:3]
        m = V.pair_metrics(k0, k1, np.arange(50), Ht, (192, 192))
        check('eval: perfect matches: precision 1 and the homography recovered', (m['precision'], m['h_ok']), (1.0, True))
        m = V.pair_metrics(k0, k1, rng.permutation(50), Ht, (192, 192))
        check('eval: shuffled matches: low precision, no homography', (m['precision'] < 0.2, m['h_ok']), (True, False))
        # export
        import torch
        cp = {'model': {'matcher.transformers.0.self_attn.Wqkv.weight': torch.ones(3, 3),
                        'matcher.log_assignment.0.matchability.bias': torch.zeros(1),
                        'extractor.net.weight': torch.ones(2)}, 'epoch': 7, 'eval': {'loss/total': 0.5},
              'conf': {'model': {'matcher': {'name': 'matchers.lightglue', 'input_dim': 128}}}}
        cpp = os.path.join(tmp, 'checkpoint_best.tar')
        torch.save(cp, cpp)
        X.export(cpp, os.path.join(tmp, 'ft_lightglue.pth'))
        sd = torch.load(os.path.join(tmp, 'ft_lightglue.pth'))
        check('export: only the matcher, prefix removed (the layout kornia / LightGlue load)',
              sorted(sd), ['log_assignment.0.matchability.bias', 'transformers.0.self_attn.Wqkv.weight'])
        cp2 = {'model': {**cp['model'], 'matcher.condition.modality.weight': torch.ones(6, 4)}, 'epoch': 3,
               'conf': {'model': {'matcher': {'name': 'gf_cond_lightglue'}, 'extractor': {'trainable': True}}}}
        torch.save(cp2, os.path.join(tmp, 'cp2.tar'))
        X.export(os.path.join(tmp, 'cp2.tar'), os.path.join(tmp, 'ex', 'c_lightglue.pth'))
        check('export: the condition and a trained extractor go to their own files; LightGlue stays plain',
              (sorted(os.listdir(os.path.join(tmp, 'ex'))), sorted(torch.load(os.path.join(tmp, 'ex', 'c_lightglue.pth'))),
               sorted(torch.load(os.path.join(tmp, 'ex', 'c_lightglue_condition.pth'))),
               sorted(torch.load(os.path.join(tmp, 'ex', 'c_extractor.pth')))),
              (['c_extractor.pth', 'c_lightglue.json', 'c_lightglue.pth', 'c_lightglue_condition.pth'],
               ['log_assignment.0.matchability.bias', 'transformers.0.self_attn.Wqkv.weight'], ['modality.weight'],
               ['net.weight']))
        # the condition: identity at the start, then per modality and pixel size
        import condition as C
        torch.manual_seed(0)
        cond = C.Condition(8)
        desc = torch.randn(2, 5, 8)
        mods, pxs = torch.tensor([4, 0]), torch.tensor([45.0, 315.0])
        same = torch.equal(cond(desc, mods, pxs), desc)
        opt = torch.optim.SGD(cond.parameters(), lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            loss = ((cond(desc, mods, pxs) - (desc * 0.5 + 0.1)) ** 2).sum()
            loss.backward()
            opt.step()
        out_c = cond(desc, mods, pxs)
        moved = (out_c - desc).abs().amax(dim=(1, 2))
        check('condition: starts as the identity (the official matcher), learns from the loss, per modality',
              (same, bool((moved > 1e-3).all()), bool(cond.modality.weight[1].abs().sum() == 0),
               bool(cond.resolution[2].weight.abs().sum() > 0)), (True, True, True, True))
        # GPU-service entry, pairs mode
        js = os.path.join(tmp, 'mode_pairs.json')
        json.dump({'mode': 'pairs', 'manifest': mp, 'crop': 96, 'per_scene': 2, 'res_m': [100, 140]}, open(js, 'w'))
        rc = Q.main(['gpuaas.py', '.', os.path.join(tmp, 'svc_out'), js])
        isv = pd.read_csv(os.path.join(tmp, 'svc_out', 'pairs', 'pairs.csv'))
        check('gpuaas: pairs mode writes the pairs into the output folder, within the given pixel-size range',
              (rc, len(isv) > 0, bool(isv['px_m'].between(100, 140).all())), (0, True, True))
        # evaluation summaries by sensor, modality pair and pixel-size band
        ev = V.summarize([{'sensor': 'MicroSat', 'modality0': 'lwir', 'modality1': 'visible', 'px_m': 45.0,
                           'matches': 10, 'correct': 5, 'h_ok': True},
                          {'sensor': 'C2E', 'modality0': 'visible', 'modality1': 'visible', 'px_m': 250.0,
                           'matches': 20, 'correct': 20, 'h_ok': True}])
        check('eval: summaries overall, per sensor, per modality pair and per pixel-size band',
              [(r['by'], r['group'], r['pairs'], r['precision']) for r in ev.to_dict('records')],
              [('all', 'all', 2, 0.8333), ('sensor', 'C2E', 1, 1.0), ('sensor', 'MicroSat', 1, 0.5),
               ('modality', 'lwir -> visible', 1, 0.5), ('modality', 'visible -> visible', 1, 1.0),
               ('resolution', '30-100 m', 1, 0.5), ('resolution', '100-300 m', 1, 1.0)])
        rc = Q.run_train({'index': 'x'}, os.path.join(tmp, 'svc_out')) if 'gluefactory' not in sys.modules else 0
        try:
            import gluefactory  # noqa: F401
            have_gf = True
        except Exception:
            have_gf = False
        if not have_gf:
            check('gpuaas: without Glue Factory, train mode says how to install it', rc, 2)
            # the command line train mode builds, with a stand-in for Glue Factory (only its settings are read)
            import types
            fake = types.ModuleType('gluefactory')
            fake.settings = types.SimpleNamespace(TRAINING_PATH=os.path.join(tmp, 'gf_outputs'))
            os.makedirs(os.path.join(tmp, 'gf_outputs', 'smoke_exp'))
            shutil.copyfile(cpp, os.path.join(tmp, 'gf_outputs', 'smoke_exp', 'checkpoint_best.tar'))
            calls, real_call = [], Q.subprocess.call
            sys.modules['gluefactory'], Q.subprocess.call = fake, (lambda a, env=None: calls.append((a, env)) or 0)
            try:
                rc = Q.run_train({'index': 'P.csv', 'config': 'aliked', 'experiment': 'smoke_exp', 'smoke': True,
                                  'sensors': ['MicroSat', 'RS2 LISS4']}, os.path.join(tmp, 'train_out'))
            finally:
                Q.subprocess.call = real_call
                del sys.modules['gluefactory']
            a, env = calls[0]
            check('gpuaas: train mode runs gluefactory.train on the GCP config, with gf_gcp importable',
                  (a[1:5], a[5] == Q.CONFIGS['aliked'], env['PYTHONPATH'].split(os.pathsep)[0]),
                  (['-m', 'gluefactory.train', 'smoke_exp', '--conf'], True, HERE))
            check('gpuaas: smoke is one epoch on 64 pairs; sensors go as a list',
                  [x for x in a if x.startswith(('train.epochs', 'data.max_pairs', 'data.sensors', 'data.index'))],
                  ['data.index=P.csv', 'train.epochs=1', 'data.max_pairs=64', 'data.sensors=["MicroSat", "RS2 LISS4"]'])
            check('gpuaas: after training, the best checkpoint and its LightGlue export are in the output folder',
                  (rc, sorted(os.listdir(os.path.join(tmp, 'train_out')))),
                  (0, ['checkpoint_best.tar', 'smoke_exp_lightglue.json', 'smoke_exp_lightglue.pth']))
            sys.modules['gluefactory'], Q.subprocess.call = fake, (lambda a, env=None: calls.append((a, env)) or 0)
            try:
                Q.run_train({'index': 'P.csv', 'modalities': ['thermal'], 'train_extractor': True},
                            os.path.join(tmp, 'train_out2'))
                Q.run_train({'index': 'P.csv', 'condition': False}, os.path.join(tmp, 'train_out3'))
            finally:
                Q.subprocess.call = real_call
                del sys.modules['gluefactory']
            a2, a3 = calls[1][0], calls[2][0]
            check('gpuaas: modalities filter, extractor training, and plain LightGlue when condition is false',
                  ('data.modalities=["thermal"]' in a2, 'model.extractor.trainable=True' in a2,
                   any(x.startswith('model.matcher.name') for x in a2), 'model.matcher.name=matchers.lightglue' in a3),
                  (True, True, False, True))
            # the conditioned matcher on a stand-in for Glue Factory's LightGlue
            import types
            from torch import nn

            class _Conf(dict):
                __getattr__ = dict.__getitem__

            class _LightGlue(nn.Module):
                default_conf = {'input_dim': 8, 'name': 'lightglue'}

                def __init__(self, conf):
                    super().__init__()
                    self.conf = _Conf({**self.default_conf, **conf})

                def forward(self, data):
                    return {'got0': data['descriptors0'], 'got1': data['descriptors1']}
            names = ['gluefactory', 'gluefactory.models', 'gluefactory.models.matchers',
                     'gluefactory.models.matchers.lightglue']
            for n in names:
                sys.modules[n] = types.ModuleType(n)
            sys.modules['gluefactory.models.matchers.lightglue'].LightGlue = _LightGlue
            try:
                sys.modules.pop('gf_cond_lightglue', None)
                import gf_cond_lightglue as GL
                mg = GL.CondLightGlue({'input_dim': 8})
                d0, d1 = torch.randn(2, 5, 8), torch.randn(2, 7, 8)
                batch = {'descriptors0': d0, 'descriptors1': d1, 'modality0': torch.tensor([4, 0]),
                         'modality1': torch.tensor([0, 0]), 'px_m': torch.tensor([45.0, 45.0], dtype=torch.float64)}
                r0 = mg(batch)
                with torch.no_grad():
                    mg.condition.modality.weight[4] = 0.5
                r1 = mg(batch)
                r2 = mg({'descriptors0': d0, 'descriptors1': d1})
                check('gf_cond_lightglue: the official matcher at the start; then the thermal view, and only it, '
                      'is transformed; without modalities it is plain LightGlue',
                      (torch.equal(r0['got0'], d0), not torch.equal(r1['got0'][0], d0[0]),
                       torch.equal(r1['got0'][1], d0[1]), torch.equal(r1['got1'], d1), torch.equal(r2['got0'], d0),
                       GL.__main_model__ is GL.CondLightGlue),
                      (True, True, True, True, True, True))
            finally:
                for n in names + ['gf_cond_lightglue']:
                    sys.modules.pop(n, None)
    finally:
        if '--keep' not in sys.argv:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(FAILS)} failure(s)' + (': ' + '; '.join(FAILS) if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
