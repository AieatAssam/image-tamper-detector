#!/usr/bin/env python3
"""Fetch and verify the optional, local-only IMD2020 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/corpus/imd2020"
ARCHIVE = CACHE / "IMD2020.zip"
EXTRACTED = CACHE / "extracted"
URL = "https://staff.utia.cas.cz/novozada/db/IMD2020.zip"
USER_AGENT = "image-tamper-detector-corpus/1.0"
CHUNK_SIZE = 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
# The host omits its issuing intermediate. This CA is the issuer advertised by
# the live certificate; certifi still validates the chain to its trusted root.
SERVER_ISSUER_CERT = """-----BEGIN CERTIFICATE-----
MIIGBTCCA+2gAwIBAgIQFNV782kiKCGaVWf6kWUbIjANBgkqhkiG9w0BAQsFADBs
MQswCQYDVQQGEwJHUjE3MDUGA1UECgwuSGVsbGVuaWMgQWNhZGVtaWMgYW5kIFJl
c2VhcmNoIEluc3RpdHV0aW9ucyBDQTEkMCIGA1UEAwwbSEFSSUNBIFRMUyBSU0Eg
Um9vdCBDQSAyMDIxMB4XDTI1MDEwMzExMTUwMFoXDTM5MTIzMTExMTQ1OVowYDEL
MAkGA1UEBhMCR1IxNzA1BgNVBAoMLkhlbGxlbmljIEFjYWRlbWljIGFuZCBSZXNl
YXJjaCBJbnN0aXR1dGlvbnMgQ0ExGDAWBgNVBAMMD0dFQU5UIFRMUyBSU0EgMTCC
AaIwDQYJKoZIhvcNAQEBBQADggGPADCCAYoCggGBAKEEaZSzEzznAPk8IEa17GSG
yJzPTj4cwRY7/vcq2BPT5+IRGxQtaCdgLXIEl2cdPdIkj2eyakFmgMjAtyeju8V8
dRayQCD/bWjJ7thDlowgLljQaXirxnYbT8bzRHAhCZqBakYgi5KWw9dANLyDHGpX
UdY259ab0lWEaFE5Uu6IzQSMJOAy4l/Twym8GUiy0qMDEBFSlm31C9BXpdHKKAlh
vIjMiKoDeTWl5vZaLB2MMRGY1yW2ftPgIP0/MkX1uFITlvHmmMTngxplH1nybEIJ
FiwHg1KiLk1TprcZgeO2gxE5Lz3wTFWrsUlAzrh5xWmscWkjNi/4BpeuiT5+NExF
czboLnXOfjuci/7bsnPi1/aZN/iKNbJRnngFoLaKVMmqCS7Xo34f+BITatryQZFE
u2oDKExQGlxDBCfYMLgLucX/onpLzUSgeQITNLx6i5tGGbUYH+9Dy3GI66L/5tPj
qzlOsydki8ZYGE5SBJeWCZ2IrhUe0WzZ2b6Zhk6JAQIDAQABo4IBLTCCASkwEgYD
VR0TAQH/BAgwBgEB/wIBADAfBgNVHSMEGDAWgBQKSCOmYKSSCjPqk1vFV+olTb0S
7jBNBggrBgEFBQcBAQRBMD8wPQYIKwYBBQUHMAKGMWh0dHA6Ly9jcnQuaGFyaWNh
LmdyL0hBUklDQS1UTFMtUm9vdC0yMDIxLVJTQS5jZXIwEQYDVR0gBAowCDAGBgRV
HSAAMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATBCBgNVHR8EOzA5MDeg
NaAzhjFodHRwOi8vY3JsLmhhcmljYS5nci9IQVJJQ0EtVExTLVJvb3QtMjAyMS1S
U0EuY3JsMB0GA1UdDgQWBBSGAXI/jKlw4jEGUxbOAV9becg8OzAOBgNVHQ8BAf8E
BAMCAYYwDQYJKoZIhvcNAQELBQADggIBABkssjQzYrOo4GMsKegaChP16yNe6Sck
cWBymM455R2rMeuQ3zlxUNOEt+KUfgueOA2urp4j6TlPbs/XxpwuN3I1f09Luk5b
+ZgRXM7obE6ZLTerVQWKoTShyl34R2XlK8pEy7+67Ht4lcJzt+K6K5gEuoPSGQDP
ef+fUfmXrFcgBMcMbtfDb9dubFKNZZxo5nAXiqhFMOIyByag3H+tOTuH8zuId9pH
RDsUpAIHJ9/W2WBfLcKav7IKRlNBRD/sPBy903J9WHPKwl8kQSDA+aa7XCYk7bJt
Eyf+7GM9F5cZ7+YyknXqnv/rtQEkTKZdQo5Us18VFe9qqj94tXbLdk7PejJYNB4O
Zlli44Ld7rtqfFlUych7gIxFOmiyxMQQYrYmUi+74lEZvfoNhuref0CupuKpz6O3
dLv6kO9T10uNdDBoBQTkge3UzHafTIe3R2o3ujXKUGPwyc9m7/FETyKLUCwSU/5O
AVOeBCU8QtkKKjM8AmbpKpe3pHWcyq3R7B3LmIALkMPTydyDfxen65IDqREbVq8N
xjhkJThUz40JqOlN6uqKqeDISj/IoucYwsqW24AlO7ZzNmohQmMi8ep23H4hBSh0
GBTe2XvkuzaNf92syK8l2HzO+13GLCjzYLTPvXTO9UpK8DGyfGZOuamuwbAnbNpE
3RfjV9IaUQGJ
-----END CERTIFICATE-----"""


def _ssl_context() -> ssl.SSLContext:
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cadata=SERVER_ISSUER_CERT)
    return context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> tuple[int, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".zip.part")
    request = Request(URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, context=_ssl_context(), timeout=120) as response, partial.open("wb") as output:
        size = 0
        digest = hashlib.sha256()
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    partial.replace(ARCHIVE)
    return size, digest.hexdigest()


def safe_extract() -> None:
    if EXTRACTED.exists():
        shutil.rmtree(EXTRACTED)
    EXTRACTED.mkdir(parents=True)
    root = EXTRACTED.resolve()
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"corrupt archive member: {bad}")
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            target = (EXTRACTED / Path(*relative.parts)).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"archive path escapes extraction directory: {member.filename}")
            archive.extract(member, EXTRACTED)


def image_files() -> list[Path]:
    return sorted(path for path in EXTRACTED.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _is_original(path: Path) -> bool:
    return path.stem.lower().endswith(("_orig", "_original"))


def _is_mask(path: Path) -> bool:
    return path.stem.lower().endswith("_mask")


def _pair_key(path: Path) -> str:
    stem = path.stem.lower()
    while True:
        for marker in ("_mask", "_original", "_orig", "_fake"):
            if stem.endswith(marker):
                stem = stem[: -len(marker)]
                break
        else:
            return stem


def verify_triples() -> dict:
    """Verify every extracted manipulated/mask/real triple without guessing labels."""
    groups = [path for path in EXTRACTED.iterdir() if path.is_dir()]
    manipulated_count = 0
    problems: list[str] = []
    for group in sorted(groups):
        files = [path for path in group.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        originals = [path for path in files if _is_original(path)]
        masks = [path for path in files if _is_mask(path)]
        manipulated = [path for path in files if not _is_original(path) and not _is_mask(path)]
        if len(originals) != 1:
            problems.append(f"{group.name}: expected one original, found {len(originals)}")
        if not manipulated:
            problems.append(f"{group.name}: no manipulated images")
        if len(masks) != len(manipulated):
            problems.append(f"{group.name}: image/mask count mismatch ({len(manipulated)} vs {len(masks)})")
        masks_by_key: dict[str, list[Path]] = {}
        for mask in masks:
            masks_by_key.setdefault(_pair_key(mask), []).append(mask)
        for image in manipulated:
            matching = masks_by_key.get(_pair_key(image), [])
            if len(matching) != 1:
                problems.append(f"{group.name}/{image.name}: expected one mask, found {len(matching)}")
                continue
            try:
                with Image.open(image) as manipulated_image, Image.open(matching[0]) as mask:
                    manipulated_image.verify()
                    mask.verify()
                    if manipulated_image.size != mask.size:
                        problems.append(f"{group.name}/{image.name}: image/mask size mismatch")
            except Exception as exc:
                problems.append(f"{group.name}/{image.name}: decode failed: {exc}")
        manipulated_count += len(manipulated)
    if problems:
        preview = "; ".join(problems[:5])
        raise RuntimeError(f"IMD2020 triple verification failed ({len(problems)} problems): {preview}")
    return {
        "source_groups": len(groups),
        "manipulated_images": manipulated_count,
        "real_counterparts": len(groups),
        "masks": manipulated_count,
        "complete_triples": manipulated_count,
    }


def inspect() -> dict:
    files = image_files()
    suffixes: dict[str, int] = {}
    top_levels: dict[str, int] = {}
    for path in files:
        suffixes[path.suffix.lower()] = suffixes.get(path.suffix.lower(), 0) + 1
        relative = path.relative_to(EXTRACTED)
        top = relative.parts[0] if len(relative.parts) > 1 else "."
        top_levels[top] = top_levels.get(top, 0) + 1
    return {
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "image_count": len(files),
        "suffixes": suffixes,
        "source_group_count": len(top_levels),
        **verify_triples(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download and extract IMD2020.zip")
    parser.add_argument("--inspect", "--check", dest="inspect", action="store_true", help="verify extracted archive inventory")
    args = parser.parse_args()
    if not args.download and not args.inspect:
        parser.error("choose --download or --inspect")
    if args.download:
        size, digest = download()
        print(json.dumps({"url": URL, "bytes": size, "sha256": digest}, sort_keys=True))
        safe_extract()
    if not EXTRACTED.is_dir():
        raise SystemExit(f"missing extracted archive: {EXTRACTED}")
    print(json.dumps(inspect(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
