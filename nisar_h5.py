#!/usr/bin/env python3
"""
Streaming reader for NISAR HDF5 products (GSLC / GCOV / RSLC).

Why this exists
---------------
ASF publishes NISAR L2 products as a single HDF5 file and nothing else -- there
is no COG. The GSLC granule that prompted this is 22 GB. Downloading that to
look at one corner reflector is absurd, so this reads the file the way a COG is
read: over HTTP range requests, pulling only the HDF5 chunks that intersect the
window you asked for.

h5py can open any seekable file-like object, so the whole trick is supplying one
backed by `Range:` requests with a block cache. HDF5's own chunked layout does
the rest.

s3:// URLs
----------
CMR advertises a direct-access s3:// link for every granule. It works only from
inside AWS us-west-2 -- ASF issues in-region-only credentials, and says so in
its own s3credentialsREADME. Passed an s3:// URL, this module uses direct S3
when it detects it is in-region, and otherwise resolves to the HTTPS URL for the
same object (verified with a HEAD) so the read still works from a laptop or an
on-prem box.

Credentials
-----------
ASF gates the data GET behind Earthdata Login. This reads them from the places
they already live and NEVER takes them on a command line, where they would end
up in your shell history and in the process table:

  1. $EARTHDATA_TOKEN  -- an Earthdata Login bearer token (preferred: scoped,
                          revocable, and not your password)
  2. ~/.netrc          -- the standard NASA/ASF mechanism:
                             machine urs.earthdata.nasa.gov
                               login YOUR_USERNAME
                               password YOUR_PASSWORD
                          chmod 600 it.

Get a token at https://urs.earthdata.nasa.gov/profile -> Generate Token.

Nothing here logs, echoes or persists a credential.
"""

from __future__ import annotations

import io
import math
import netrc
import os
import re
import time
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

URS_HOST = "urs.earthdata.nasa.gov"


# =============================================================================
# Authenticated, seekable HTTP file object
# =============================================================================
class _EarthdataSession:
    """requests.Session that keeps Basic auth across the URS redirect chain.

    Earthdata bounces a data GET through urs.earthdata.nasa.gov and back out to
    a signed CloudFront URL. requests drops the Authorization header on a
    cross-host redirect (rightly), which breaks that dance, so we re-attach it
    for the URS host only -- and never for the signed CDN URL, which must not
    see your credentials.
    """

    def __new__(cls, token: Optional[str], basic: Optional[Tuple[str, str]]):
        import requests

        class Session(requests.Session):
            def rebuild_auth(self, prepared_request, response):
                headers = prepared_request.headers
                orig = urlparse(response.request.url).hostname
                dest = urlparse(prepared_request.url).hostname
                if "Authorization" in headers and orig != dest:
                    if URS_HOST not in (orig, dest):
                        del headers["Authorization"]

        s = Session()
        s.trust_env = True
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        elif basic:
            s.auth = basic
        return s


def _find_credentials(url: str, allow_netrc: bool = True
                      ) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """Locate an Earthdata credential without ever surfacing its value."""
    token = os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EDL_TOKEN")
    if token:
        return token.strip(), None
    if allow_netrc:
        for path in (os.environ.get("NETRC"), os.path.expanduser("~/.netrc"),
                     os.path.expanduser("~/_netrc")):
            if not path or not os.path.exists(path):
                continue
            try:
                auth = netrc.netrc(path).authenticators(URS_HOST)
            except Exception:
                continue
            if auth and auth[0] and auth[2]:
                return None, (auth[0], auth[2])
    return None, None


class _BlockCachedFile(io.RawIOBase):
    """Seekable file-like object with an LRU block cache over ranged reads.

    HDF5 issues many small, scattered metadata reads. Serving them from cached
    blocks collapses those into a handful of round trips, which is the whole
    reason streaming a 22 GB file is practical. Subclasses supply the transport
    by implementing `_fetch_range`.
    """

    block = 1 << 20
    max_blocks = 512

    def _init_cache(self):
        self._lock = threading.Lock()
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self.n_requests = 0
        self.n_bytes = 0
        self._pos = 0

    def _fetch_range(self, lo: int, hi: int) -> bytes:
        raise NotImplementedError

    def _block_at(self, idx: int) -> bytes:
        with self._lock:
            hit = self._cache.get(idx)
            if hit is not None:
                self._cache.move_to_end(idx)
                return hit
        lo = idx * self.block
        hi = min(lo + self.block, self.size) - 1
        data = self._fetch_range(lo, hi)
        with self._lock:
            self._cache[idx] = data
            self._cache.move_to_end(idx)
            while len(self._cache) > self.max_blocks:
                self._cache.popitem(last=False)
            self.n_requests += 1
            self.n_bytes += len(data)
        return data

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False
    def tell(self) -> int: return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        got = 0
        while got < n:
            idx = (self._pos + got) // self.block
            blk = self._block_at(idx)
            off = (self._pos + got) - idx * self.block
            take = min(n - got, len(blk) - off)
            if take <= 0:
                break
            buf[got:got + take] = blk[off:off + take]
            got += take
        self._pos += got
        return got

    @property
    def stats(self) -> Dict:
        return {"requests": self.n_requests, "bytes": self.n_bytes,
                "file_size": self.size}


class HttpRangeFile(_BlockCachedFile):
    """Block-cached reads over HTTP `Range:` requests.

    `stats` reports what it actually cost, which is the number worth watching
    on a metered or slow link.
    """

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 512,
                 timeout: int = 120, session=None):
        import requests  # noqa: F401  (import here to keep the module optional)

        self.url = url
        self.block = int(block)
        self.max_blocks = int(max_blocks)
        self.timeout = timeout
        self._init_cache()

        if session is not None:
            self.s = session
        else:
            token, basic = _find_credentials(url)
            self.s = _EarthdataSession(token, basic)

        try:
            r = self.s.head(url, allow_redirects=True, timeout=timeout)
        except Exception as exc:
            raise OSError(
                f"cannot reach {url}\n"
                f"  {type(exc).__name__}: {exc}\n"
                "  Check the URL, your network, and any proxy settings "
                "(HTTPS_PROXY / NO_PROXY).") from None
        if r.status_code in (401, 403):
            raise PermissionError(_auth_hint(url, r.status_code))
        r.raise_for_status()
        length = r.headers.get("Content-Length")
        if not length:
            raise OSError(f"{url}: server did not report a Content-Length; "
                          "cannot range-read it")
        self.size = int(length)

        # A server that ignores Range would silently return the whole 22 GB.
        probe = self.s.get(url, headers={"Range": "bytes=0-1"},
                           allow_redirects=True, timeout=timeout)
        if probe.status_code in (401, 403):
            raise PermissionError(_auth_hint(url, probe.status_code))
        probe.raise_for_status()
        if probe.status_code != 206 or len(probe.content) != 2:
            raise OSError(
                f"{url}: server does not honour Range requests "
                f"(status {probe.status_code}, {len(probe.content)} bytes for a "
                "2-byte request). Streaming this file is not possible; it would "
                "have to be downloaded.")
        self.n_requests += 1
        self.n_bytes += len(probe.content)

    def _fetch_range(self, lo: int, hi: int) -> bytes:
        r = self.s.get(self.url, headers={"Range": f"bytes={lo}-{hi}"},
                       allow_redirects=True, timeout=self.timeout)
        if r.status_code in (401, 403):
            raise PermissionError(_auth_hint(self.url, r.status_code))
        r.raise_for_status()
        return r.content


def _auth_hint(url: str, code: int) -> str:
    return (
        f"Earthdata Login required for {url} (HTTP {code}).\n"
        "This is expected: ASF gates NISAR data behind Earthdata Login.\n"
        "Set one of these up on YOUR machine -- do not paste credentials into a\n"
        "command line, a chat, or a shared terminal:\n"
        "  1. A bearer token (preferred -- scoped and revocable):\n"
        "       https://urs.earthdata.nasa.gov/profile -> Generate Token\n"
        "       export EARTHDATA_TOKEN='...'\n"
        "  2. Or ~/.netrc (the standard NASA/ASF mechanism):\n"
        f"       machine {URS_HOST}\n"
        "         login YOUR_USERNAME\n"
        "         password YOUR_PASSWORD\n"
        "       chmod 600 ~/.netrc\n"
        "You must also have accepted the NISAR EULA once, by downloading any\n"
        "granule through the Earthdata web UI while logged in."
    )


# =============================================================================
# Earthdata S3 direct access
# =============================================================================
# ASF's own s3credentialsREADME is explicit about the catch:
#
#   "the credentials are only valid for in-region requests, so using them with
#    your AWS CLI will not work! You must make your requests from an AWS service
#    such as Lambda or EC2 in the same region as the source bucket"
#
# So an s3:// URL is the fast path from inside us-west-2 and useless outside it.
# Rather than fail, we detect which situation we are in and fall back to the
# HTTPS URL for the same object, which works from anywhere.
S3_REGION = "us-west-2"

# Bucket -> (HTTPS host, path prefix). ASF splits products and browse imagery
# into sibling buckets that map onto sibling prefixes on the same host.
ASF_S3_TO_HTTPS = {
    "sds-n-cumulus-prod-nisar-products": ("nisar.asf.earthdatacloud.nasa.gov", "NISAR"),
    "sds-n-cumulus-prod-nisar-browse": ("nisar.asf.earthdatacloud.nasa.gov", "BROWSE"),
}


def split_s3(uri: str) -> Tuple[str, str]:
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def in_aws_region(region: str = S3_REGION, timeout: float = 0.4) -> bool:
    """Are we running inside `region`? Checked via EC2 instance metadata.

    Deliberately short-timeout: off EC2 there is nothing at 169.254.169.254 and
    we do not want to stall the CLI waiting to find that out.
    """
    env = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if env:
        return env.strip() == region
    try:
        import requests
        tok = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=timeout)
        headers = ({"X-aws-ec2-metadata-token": tok.text}
                   if tok.status_code == 200 else {})
        r = requests.get("http://169.254.169.254/latest/meta-data/placement/region",
                         headers=headers, timeout=timeout)
        return r.status_code == 200 and r.text.strip() == region
    except Exception:
        return False


def s3_to_https(uri: str, session=None, timeout: int = 60) -> Optional[str]:
    """Map an Earthdata s3:// URL to its HTTPS equivalent, verified by a HEAD.

    The mapping is a convention, not a guarantee, so the translation is checked
    before it is handed back -- a wrong guess would otherwise surface much later
    as a confusing 404.
    """
    bucket, key = split_s3(uri)
    entry = ASF_S3_TO_HTTPS.get(bucket)
    if not entry or not key:
        return None
    host, prefix = entry
    url = f"https://{host}/{prefix}/{key}"
    try:
        import requests
        s = session or requests.Session()
        r = s.head(url, allow_redirects=True, timeout=timeout)
        # 401/403 means it is there but gated -- the URL itself is right.
        if r.status_code < 400 or r.status_code in (401, 403):
            return url
    except Exception:
        return None
    return None


def _daac_s3_credentials(host: str, session, timeout: int = 60) -> Dict:
    """Fetch 1-hour temporary S3 credentials from a DAAC /s3credentials endpoint."""
    url = f"https://{host}/s3credentials"
    r = session.get(url, allow_redirects=True, timeout=timeout)
    if r.status_code in (401, 403):
        raise PermissionError(_auth_hint(url, r.status_code))
    r.raise_for_status()
    creds = r.json()
    missing = [k for k in ("accessKeyId", "secretAccessKey", "sessionToken")
               if k not in creds]
    if missing:
        raise OSError(f"{url}: credential response missing {missing}")
    return creds


class S3RangeFile(_BlockCachedFile):
    """Block-cached ranged reads straight from S3, for in-region use.

    Credentials last an hour, so they are refreshed on expiry rather than
    fetched once -- a long session would otherwise die partway through.
    """

    def __init__(self, uri: str, block: int = 1 << 20, max_blocks: int = 512,
                 region: str = S3_REGION, timeout: int = 120):
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise OSError(
                "boto3 is required for s3:// direct access "
                f"(pip install boto3). Import error: {exc}\n"
                "  Or just use the https:// URL for the same object, which "
                "works from anywhere.") from None

        self.uri = uri
        self.bucket, self.key = split_s3(uri)
        self.block = int(block)
        self.max_blocks = int(max_blocks)
        self.region = region
        self.timeout = timeout
        self._init_cache()

        # The DAAC that fronts this bucket is also the one that issues its
        # temporary credentials.
        host = ASF_S3_TO_HTTPS.get(
            self.bucket, ("nisar.asf.earthdatacloud.nasa.gov",))[0]
        token, basic = _find_credentials(uri)
        self._edl = _EarthdataSession(token, basic)
        self._cred_host = host
        self._client = None
        self._expiry = 0.0
        self._refresh()

        head = self._client.head_object(Bucket=self.bucket, Key=self.key)
        self.size = int(head["ContentLength"])

    def _refresh(self):
        import boto3
        creds = _daac_s3_credentials(self._cred_host, self._edl, self.timeout)
        self._client = boto3.client(
            "s3", region_name=self.region,
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"])
        # Renew a few minutes early rather than racing the expiry.
        self._expiry = time.time() + 50 * 60

    def _fetch_range(self, lo: int, hi: int) -> bytes:
        if time.time() > self._expiry:
            self._refresh()
        r = self._client.get_object(Bucket=self.bucket, Key=self.key,
                                    Range=f"bytes={lo}-{hi}")
        return r["Body"].read()


def resolve_uri(uri: str, verbose: bool = True) -> Tuple[str, str]:
    """Decide how to read `uri`. Returns (mode, resolved_uri).

    mode is "s3" (direct, in-region), "http", or "local".
    """
    if not uri.startswith("s3://"):
        return ("http" if uri.startswith(("http://", "https://")) else "local"), uri

    if in_aws_region():
        return "s3", uri

    https = s3_to_https(uri)
    if https:
        if verbose:
            print("note: s3:// direct access only works from inside AWS "
                  f"{S3_REGION} (ASF issues in-region-only credentials).\n"
                  "      Falling back to the HTTPS URL for the same object:\n"
                  f"      {https}")
        return "http", https

    raise SystemExit(
        f"cannot read {uri}\n"
        f"  Earthdata s3:// access needs to run inside AWS {S3_REGION} -- ASF's\n"
        "  temporary credentials are in-region only, so this fails from a\n"
        "  laptop, an on-prem box, or the AWS CLI anywhere else.\n"
        "  No HTTPS equivalent could be derived for this bucket either.\n"
        "  Use the https:// download URL instead -- `cog_locate.py find` and\n"
        "  CMR both give it directly.")


def open_h5(uri: str, block: int = 1 << 20):
    """Open a NISAR HDF5, local or remote. Returns (h5py.File, backing_or_None)."""
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "h5py is required to read NISAR HDF5 products.\n"
            f"  pip install h5py        (import error: {exc})")

    mode, resolved = resolve_uri(uri)
    if mode == "local":
        return h5py.File(resolved, "r"), None

    backing = (S3RangeFile(resolved, block=block) if mode == "s3"
               else HttpRangeFile(resolved, block=block))
    # A generous chunk cache pays for itself when neighbouring image chunks
    # share HDF5 metadata blocks.
    return h5py.File(backing, "r", rdcc_nbytes=128 * 1024 * 1024), backing


# =============================================================================
# NISAR product structure
# =============================================================================
# Products differ in where the imagery lives, but all follow
# /science/<L|S>SAR/<PRODUCT>/grids|swaths/frequency<A|B>/<POL>.
_GRID_RE = re.compile(
    r"^/science/(?P<band>[LS]SAR)/(?P<product>[A-Z]+)/(?:grids|swaths)"
    r"(?:/frequency(?P<freq>[AB]))?")

POL_NAMES = ("HH", "HV", "VH", "VV", "RH", "RV",
             "HHHH", "HVHV", "VHVH", "VVVV", "HHHV", "HHVV", "HVVV")


def describe(h5) -> Dict:
    """Inventory a NISAR file: band, product, frequencies, polarizations, grid."""
    import h5py

    out: Dict = {"band": None, "product": None, "frequencies": {}}

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        path = "/" + name
        m = _GRID_RE.match(path)
        if not m:
            return
        leaf = path.rsplit("/", 1)[-1]
        freq = m.group("freq")
        out["band"] = out["band"] or m.group("band")
        out["product"] = out["product"] or m.group("product")
        if freq is None:
            return
        f = out["frequencies"].setdefault(
            freq, {"pols": {}, "x": None, "y": None, "epsg": None, "extra": {}})
        if leaf in POL_NAMES and obj.ndim == 2:
            f["pols"][leaf] = {"path": path, "shape": tuple(obj.shape),
                               "dtype": str(obj.dtype),
                               "chunks": tuple(obj.chunks) if obj.chunks else None,
                               "complex": _is_complex(obj.dtype)}
        elif leaf in ("xCoordinates", "xCoordinateSpacing"):
            f["x"] = f["x"] or (path if leaf == "xCoordinates" else None)
        elif leaf in ("yCoordinates", "yCoordinateSpacing"):
            f["y"] = f["y"] or (path if leaf == "yCoordinates" else None)
        elif leaf == "projection":
            try:
                f["epsg"] = int(np.asarray(obj[()]).ravel()[0])
            except Exception:
                for key in ("epsg_code", "EPSG", "spatial_ref"):
                    if key in obj.attrs:
                        try:
                            f["epsg"] = int(np.asarray(obj.attrs[key]).ravel()[0])
                        except Exception:
                            pass

    h5.visititems(visit)
    return out


def _is_complex(dtype) -> bool:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        return True
    # NISAR GSLC often stores complex as a compound {r, i} pair.
    return bool(dt.names) and set(n.lower() for n in dt.names) in (
        {"r", "i"}, {"real", "imag"}, {"re", "im"})


def _to_complex(arr: np.ndarray) -> np.ndarray:
    dt = arr.dtype
    if np.issubdtype(dt, np.complexfloating):
        return arr
    if dt.names:
        names = list(dt.names)
        return (arr[names[0]].astype(np.float32)
                + 1j * arr[names[1]].astype(np.float32))
    return arr


def read_window(h5, path: str, row0: int, col0: int, nrow: int, ncol: int,
                dec: int = 1) -> np.ndarray:
    """Read a window, converting complex to intensity |z|^2.

    GSLC is complex: the meaningful display and peak-location quantity is power,
    not the real part. Decimation is a strided read, which is what keeps a
    whole-scene overview from pulling the entire array over the wire.
    """
    ds = h5[path]
    H, W = ds.shape[0], ds.shape[1]
    r0 = max(0, min(row0, H))
    c0 = max(0, min(col0, W))
    r1 = max(r0, min(row0 + nrow, H))
    c1 = max(c0, min(col0 + ncol, W))
    if r1 <= r0 or c1 <= c0:
        raise SystemExit(
            f"requested window is entirely outside {path} "
            f"(array is {H} x {W} px; asked for rows {row0}..{row0+nrow}, "
            f"cols {col0}..{col0+ncol})")

    raw = ds[r0:r1:dec, c0:c1:dec]
    arr = _to_complex(np.asarray(raw))
    if np.iscomplexobj(arr):
        out = (arr.real.astype(np.float32) ** 2 + arr.imag.astype(np.float32) ** 2)
    else:
        out = arr.astype(np.float32)
    return out, (r0, c0)


def grid_transform(h5, freq_info: Dict):
    """Affine transform + CRS from the product's coordinate arrays.

    NISAR stores pixel-CENTRE coordinates, so the transform's origin is shifted
    back by half a pixel to the corner convention GDAL and this tool use.
    """
    from affine import Affine

    xp, yp = freq_info.get("x"), freq_info.get("y")
    if not xp or not yp:
        raise SystemExit("product has no xCoordinates/yCoordinates arrays; "
                         "cannot georeference it")
    x = np.asarray(h5[xp][()], dtype=np.float64)
    y = np.asarray(h5[yp][()], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise SystemExit("coordinate arrays too short to derive a pixel size")

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    tf = Affine(dx, 0.0, float(x[0]) - dx / 2.0,
                0.0, dy, float(y[0]) - dy / 2.0)

    crs = None
    epsg = freq_info.get("epsg")
    if epsg:
        try:
            from rasterio.crs import CRS
            crs = CRS.from_epsg(int(epsg))
        except Exception:
            crs = None
    return tf, crs, (dx, dy)
