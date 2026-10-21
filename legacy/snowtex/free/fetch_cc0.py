#!/usr/bin/env python3
"""
Fetch the CC0 snow / track / pavement texture set (Poly Haven + ambientCG) at 2K.

Run on any machine with internet access:
    python3 fetch_cc0.py            # downloads into ./cc0/ next to this script
Standard library only. Safe to re-run: finished items are skipped.

Keeps per item: colour (albedo), displacement/height, normal (OpenGL). Everything else
in the source packs (roughness, AO, previews) is dropped to save space.
Writes manifest.csv: source, id, licence, source page, file, bytes, sha256.
"""
import csv, hashlib, io, json, os, shutil, ssl, subprocess, sys, time, urllib.request, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cc0")
UA = {"User-Agent": "IntelSet-snow-texture-fetch/1.0 (research dataset; contact bykov2004@gmail.com)"}

POLYHAVEN = [
    # snow
    "snow_01", "snow_02", "snow_03", "snow_04", "snow_05", "snow_floor", "asphalt_snow", "snow_field_aerial",
    # tyre tracks / tread donors
    "muddy_tracks", "aerial_mud_1", "moon_track_01", "moon_track_02", "moon_track_03", "moon_track_04",
    "red_sand", "park_sand", "dirt_aerial_02",
    # pavement bases / micro-relief
    "asphalt_01", "asphalt_02", "asphalt_03", "asphalt_04", "asphalt_05", "asphalt_06", "asphalt_07",
    "clean_asphalt", "road_damaged", "road_damaged_2", "road_damaged_clean", "tarred_gravel", "worn_asphalt",
    "asphalt_floor", "aerial_asphalt_01", "concrete_pavers_02", "square_tiles_03", "bitumen",
]
AMBIENTCG = [
    # snow
    "Snow001", "Snow002", "Snow003", "Snow004", "Snow005", "Snow006", "Snow007A", "Snow007B", "Snow007C",
    "Snow008A", "Snow008B", "Snow008C", "Snow009A", "Snow009B", "Snow009C", "Snow010A", "Snow010B", "Snow010C",
    "Snow011", "Snow012", "Snow013", "Snow014", "Snow015",
    # footprints / tracks
    "Footsteps001", "Footsteps003", "Footsteps004", "Footsteps005", "Footsteps006", "Ground017",
    # wet / puddles / ice
    "Asphalt025B", "Asphalt025C", "Asphalt024C", "Ground050", "PavingStones115A", "Ice001", "Ice002",
    # pavement bases
    "Road001", "Road002", "Road003", "Road006", "Road007", "Road008A", "Road009A", "Road010A", "Road011A",
    "Road012A", "Road013A", "Road014A", "Road015A", "Asphalt010", "Asphalt015", "Asphalt020S", "Asphalt023S",
    "Asphalt031", "Asphalt033",
]

# map-name candidates per source, in order of preference
PH_MAPS = {"color": [("Diffuse", "jpg"), ("diff", "jpg")],
           "height": [("Displacement", "png"), ("disp", "png"), ("Displacement", "jpg")],
           "normal": [("nor_gl", "jpg"), ("nor_gl", "png")]}
AC_KEEP = {"color": "_Color.", "height": "_Displacement.", "normal": "_NormalGL."}


def _curl(url):
    """macOS system curl uses the Keychain, so it works when Python has no CA bundle."""
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-A", UA["User-Agent"], url], capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode}: {r.stderr.decode(errors='ignore')[:200]}")
    return r.stdout


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


CTX = _ssl_ctx()
# macOS: system curl is the reliable path (Python from python.org ships without CA certificates)
USE_CURL = sys.platform == "darwin" and shutil.which("curl") is not None
VERSION = "fetch_cc0 v3 (curl on macOS, urllib elsewhere)"


def get(url, tries=4):
    global USE_CURL
    if USE_CURL:
        return _curl(url)
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120, context=CTX) as r:
                return r.read()
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e) and shutil.which("curl"):
                USE_CURL = True                      # switch once, for the rest of the run
                print("  (Python has no CA certificates - using system curl from now on)", flush=True)
                return _curl(url)
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(path, data, rows, src, aid, page):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    rows.append([src, aid, "CC0", page, os.path.relpath(path, ROOT), len(data), sha(data)])


def polyhaven(aid, rows):
    out = os.path.join(ROOT, "polyhaven", aid)
    if os.path.exists(os.path.join(out, ".done")):
        return "skip"
    files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
    got = []
    for role, cands in PH_MAPS.items():
        for key, ext in cands:
            try:
                url = files[key]["2k"][ext]["url"]
            except (KeyError, TypeError):
                continue
            data = get(url)
            save(os.path.join(out, f"{aid}_{role}_2k.{ext}"), data, rows, "polyhaven", aid, f"https://polyhaven.com/a/{aid}")
            got.append(role)
            break
    open(os.path.join(out, ".done"), "w").write(",".join(got))
    return ",".join(got)


def ambientcg(aid, rows):
    out = os.path.join(ROOT, "ambientcg", aid)
    if os.path.exists(os.path.join(out, ".done")):
        return "skip"
    data = get(f"https://ambientcg.com/get?file={aid}_2K-JPG.zip")
    got = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            for role, tag in AC_KEEP.items():
                if tag in name and not name.lower().endswith(".usdc"):
                    b = z.read(name)
                    save(os.path.join(out, f"{aid}_{role}_2k" + os.path.splitext(name)[1]), b, rows, "ambientcg", aid,
                         f"https://ambientcg.com/view?id={aid}")
                    got.append(role)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, ".done"), "w").write(",".join(got))
    return ",".join(got)


def main():
    print(VERSION, "| downloader:", "curl" if USE_CURL else "urllib", flush=True)
    os.makedirs(ROOT, exist_ok=True)
    rows, failed = [], []
    jobs = [("polyhaven", a, polyhaven) for a in POLYHAVEN] + [("ambientcg", a, ambientcg) for a in AMBIENTCG]
    for i, (src, aid, fn) in enumerate(jobs, 1):
        try:
            r = fn(aid, rows)
            print(f"[{i:3d}/{len(jobs)}] {src:9s} {aid:22s} {r}", flush=True)
        except Exception as e:
            failed.append((src, aid, str(e)))
            print(f"[{i:3d}/{len(jobs)}] {src:9s} {aid:22s} FAILED: {e}", flush=True)
        time.sleep(0.5)
    man = os.path.join(ROOT, "manifest.csv")
    new = not os.path.exists(man)
    with open(man, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "id", "licence", "source_page", "file", "bytes", "sha256"])
        w.writerows(rows)
    print(f"\nnew files: {len(rows)}, failed items: {len(failed)}")
    for f in failed:
        print("  failed:", *f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
