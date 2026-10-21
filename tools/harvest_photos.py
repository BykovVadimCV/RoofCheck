#!/usr/bin/env python3
"""
M1 harvest_sources: openly licensed winter photos from Openverse (Flickr, Wikimedia, ...) and
Wikimedia Commons, for the snow texture bank (M8) and as real winter references.

    python3 harvest_photos.py                 # CC0 + Public Domain + CC BY
    python3 harvest_photos.py --by-sa         # also CC BY-SA (share-alike, flagged in manifest)
    python3 harvest_photos.py --dry-run       # metadata only -> candidates.csv, no image downloads
    python3 harvest_photos.py --max-per-query 100 --max-total 3000

Output (next to this script):  photos/<source>/<id>.jpg  +  photos/manifest.csv
Standard library only; Pillow is used if installed (near-duplicate removal + resize to 2048 px).
Safe to re-run: items already in the manifest are skipped; Openverse paging resumes from state.json.

Openverse anonymous limits: 240 results per query, ~20 requests/min, ~200 requests/day. The script
waits on the per-minute limit and stops cleanly on the daily one; run it again the next day, or
register a free Openverse API client and set OPENVERSE_CLIENT_ID / OPENVERSE_CLIENT_SECRET.

Licences kept: cc0, pdm (public domain), by; by-sa only with --by-sa. NC / ND / GFDL-only are never kept.
Every row is marked needs_privacy_blur=1: run the face / plate blur (M3) before anything else uses it.
"""
import argparse, csv, hashlib, html, json, os, re, shutil, ssl, subprocess, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "photos")
MANIFEST = os.path.join(OUT, "manifest.csv")
STATE = os.path.join(OUT, "state.json")
UA = "IntelSet-snow-harvest/1.0 (snow-clearance CV research; contact bykov2004@gmail.com)"
VERSION = "harvest_photos v1.2"

QUERIES = [
    # ground snow in urban settings
    "snow parking lot", "snowy parking lot", "parking lot snow cars", "snow covered parking",
    "snow covered road", "snowy street", "snow city street", "residential street snow",
    "snow sidewalk", "icy sidewalk", "snow covered sidewalk", "snowy courtyard",
    # tracks / traffic
    "tire tracks snow", "tyre tracks snow", "car tracks snow", "ruts snow road", "packed snow road",
    "slush road", "slush street", "dirty snow", "black snow city", "melting snow street",
    # clearing
    "snow plow", "snow plowing parking lot", "snow removal", "plowed road snow", "snowbank",
    "snow pile parking lot", "snow pile city", "snow clearing tractor", "shoveling snow",
    # surfaces
    "snow asphalt", "icy road", "frozen puddle road", "frost asphalt", "snow pavement",
    # roofs
    "snow roof", "snow covered roof", "flat roof snow", "roof snow removal",
]
# Commons also indexes Russian descriptions
COMMONS_EXTRA = ["снег парковка", "сугроб", "уборка снега", "колея снег", "снег двор", "гололёд"]

# --set roofs: roofs seen from a taller building, roofs with snow, roof cleaning, icicles
ROOF_QUERIES = [
    "snow covered roofs", "snowy rooftops", "rooftops winter", "roofs snow city", "view over rooftops winter",
    "rooftop view snow", "city roofs winter view", "roof after snowfall", "snow on metal roof",
    "standing seam roof snow", "tin roof snow", "pitched roof snow", "snow drift roof", "chimney snow roof",
    "roof snow cornice", "snow overhang roof edge", "snow melting roof", "roof snow cleaning",
    "clearing snow from roof", "roof snow shoveling", "workers removing snow roof", "icicles roof",
    "icicles eaves", "Moscow rooftops winter", "Moscow roofs snow", "Saint Petersburg roofs winter",
    "Riga roofs winter", "Prague roofs snow", "rooftops Moscow", "metal roofs city",
]
ROOF_COMMONS_EXTRA = ["крыши зимой", "снег на крыше", "крыши Москвы", "вид на крыши", "уборка снега с крыши",
                      "очистка кровли от снега", "сосульки", "кровля снег", "крыши Петербурга зимой"]

ALLOWED = {"cc0", "pdm", "by"}


# ---------------------------------------------------------------- http
def _curl(url, headers=None, data=None):
    cmd = ["curl", "-fsSL", "--retry", "3", "-A", UA]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    if data is not None:
        cmd += ["--data", data]
    r = subprocess.run(cmd + [url], capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode}: {r.stderr.decode(errors='ignore')[:200]}")
    return r.stdout


USE_CURL = sys.platform == "darwin" and shutil.which("curl") is not None   # macOS Python often lacks CA certs


def http(url, headers=None, data=None, want_headers=False):
    """Returns body (and headers if asked). curl path has no response headers -> {}."""
    h = {"User-Agent": UA, **(headers or {})}
    if USE_CURL and not want_headers:
        return _curl(url, headers, data)
    req = urllib.request.Request(url, headers=h, data=data.encode() if data else None)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            return (body, dict(r.headers), r.status) if want_headers else body
    except urllib.error.HTTPError as e:
        if want_headers:
            return e.read(), dict(e.headers), e.code
        raise
    except Exception as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e) and shutil.which("curl"):
            body = _curl(url, headers, data)
            return (body, {}, 200) if want_headers else body
        raise


def http_json_with_status(url, headers=None):
    """JSON + status + headers, via curl -i on macOS so rate-limit headers stay visible."""
    if USE_CURL:
        cmd = ["curl", "-sS", "-i", "-A", UA] + sum([["-H", f"{k}: {v}"] for k, v in (headers or {}).items()], []) + [url]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        raw = r.stdout.decode("utf-8", errors="ignore")
        # last header block (curl may print 100-continue etc.)
        parts = re.split(r"\r?\n\r?\n", raw, maxsplit=1)
        head, body = (parts + [""])[:2]
        status = int(re.search(r"HTTP/\S+\s+(\d+)", head).group(1)) if re.search(r"HTTP/\S+\s+(\d+)", head) else 0
        hdrs = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in head.splitlines()[1:] if ":" in l)}
        return status, hdrs, body
    body, hdrs, status = http(url, headers, want_headers=True)
    return status, {k.lower(): v for k, v in hdrs.items()}, body.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- state / manifest
FIELDS = ["source", "provider", "id", "query", "title", "creator", "creator_url", "license", "license_version",
          "license_url", "share_alike", "landing_url", "image_url", "width", "height", "file", "bytes", "sha256",
          "dhash", "needs_privacy_blur", "harvested_at"]


def load_manifest():
    seen, hashes = set(), []
    if os.path.exists(MANIFEST):
        for r in csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")):
            seen.add((r["source"], r["id"]))
            if r.get("dhash"):
                hashes.append(int(r["dhash"], 16))
    return seen, hashes


def append_rows(rows):
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"openverse_pages": {}, "commons_offsets": {}}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)


# ---------------------------------------------------------------- image handling
try:
    from PIL import Image
    import io
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def dhash(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("L").resize((9, 8))
    px = im.tobytes()                                        # 72 grey values, works on all Pillow versions
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def shrink(img_bytes, max_side=2048):
    im = Image.open(io.BytesIO(img_bytes))
    if max(im.size) <= max_side:
        return img_bytes, im.size
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue(), im.size


def store(item, seen, hashes, args):
    """Download one candidate; returns manifest row or None."""
    key = (item["source"], item["id"])
    if key in seen:
        return None
    data = http(item["image_url"])
    if len(data) < 20_000:                                  # thumbnails / broken files
        return None
    w, h = item.get("width"), item.get("height")
    dh = ""
    if HAVE_PIL:
        try:
            d = dhash(data)
        except Exception:
            return None                                     # not a decodable image
        if any(bin(d ^ o).count("1") <= args.dup_bits for o in hashes):
            return None                                     # near-duplicate of something we have
        hashes.append(d)
        dh = f"{d:016x}"
        data, (w, h) = shrink(data)
        if min(w, h) < args.min_side * 0.75:                # provider served a small rendition
            hashes.pop()
            return None
    fn = os.path.join(OUT, item["source"], re.sub(r"[^A-Za-z0-9_.-]", "_", item["id"])[:120] + ".jpg")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    open(fn, "wb").write(data)
    seen.add(key)
    return {**{k: item.get(k, "") for k in FIELDS}, "width": w, "height": h, "file": os.path.relpath(fn, OUT),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dhash": dh,
            "needs_privacy_blur": 1, "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ---------------------------------------------------------------- Openverse
def openverse_token():
    cid, sec = os.environ.get("OPENVERSE_CLIENT_ID"), os.environ.get("OPENVERSE_CLIENT_SECRET")
    if not (cid and sec):
        return None
    body = urllib.parse.urlencode({"client_id": cid, "client_secret": sec, "grant_type": "client_credentials"})
    j = json.loads(http("https://api.openverse.org/v1/auth_tokens/token/", data=body))
    return j["access_token"]


def openverse(args, state, seen, hashes):
    lic = ",".join(sorted(ALLOWED))
    tok = openverse_token()
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    max_pages = 50 if tok else 12                         # anonymous depth limit = 240 results
    print(f"Openverse: {'registered' if tok else 'anonymous'} access, licences {lic}", flush=True)
    total = 0
    for q in QUERIES:
        page = state["openverse_pages"].get(q, 0)
        got_q = 0
        while page < max_pages and got_q < args.max_per_query and total < args.max_total:
            page += 1
            url = ("https://api.openverse.org/v1/images/?" +
                   urllib.parse.urlencode({"q": q, "license": lic, "page_size": 20, "page": page, "mature": "false"}))
            status, hd, body = http_json_with_status(url, hdr)
            if status == 429:
                wait = int(hd.get("retry-after", "60") or 60)
                if wait > 600:
                    print(f"  Openverse daily limit reached; run again later (state saved).", flush=True)
                    save_state(state)
                    return total
                print(f"  rate limit, waiting {wait}s", flush=True)
                time.sleep(wait)
                page -= 1
                continue
            if status in (400, 401, 404):                     # past last page / depth limit
                break
            if status != 200:
                print(f"  {q!r} page {page}: HTTP {status}", flush=True)
                break
            res = json.loads(body).get("results", [])
            if not res:
                break
            rows = []
            for r in res:
                if r.get("license") not in ALLOWED:
                    continue
                if min(r.get("width") or 0, r.get("height") or 0) < args.min_side:
                    continue
                item = dict(source="openverse", provider=r.get("provider") or r.get("source"), id=r["id"], query=q,
                            title=r.get("title") or "", creator=r.get("creator") or "", creator_url=r.get("creator_url") or "",
                            license=r["license"], license_version=r.get("license_version") or "",
                            license_url=r.get("license_url") or "", share_alike=int(r["license"] == "by-sa"),
                            landing_url=r.get("foreign_landing_url") or "", image_url=r["url"],
                            width=r.get("width"), height=r.get("height"))
                if args.dry_run:
                    rows.append({**{k: item.get(k, "") for k in FIELDS}})
                    continue
                try:
                    row = store(item, seen, hashes, args)
                except Exception as e:
                    print(f"    skip {r['id']}: {e}", flush=True)
                    continue
                if row:
                    rows.append(row)
            if args.dry_run:
                write_candidates(rows)
            else:
                append_rows(rows)
            got_q += len(rows); total += len(rows)
            state["openverse_pages"][q] = page
            save_state(state)
            time.sleep(3.2 if not tok else 0.5)               # stay under ~20 req/min anonymous
        print(f"  openverse {q!r}: +{got_q} (total {total})", flush=True)
    return total


# ---------------------------------------------------------------- Wikimedia Commons
def commons_licence(meta):
    s = " ".join(str(meta.get(k, {}).get("value", "")) for k in ("License", "LicenseShortName", "UsageTerms")).lower()
    if re.search(r"\bnc\b|non-?commercial|\bnd\b|noderiv", s):
        return None
    if "cc0" in s or "cc-zero" in s:
        return "cc0"
    if re.search(r"public domain|\bpd\b|pd-", s):
        return "pdm"
    if re.search(r"cc[- ]by[- ]sa", s):
        return "by-sa"
    if re.search(r"cc[- ]by\b|cc-by-\d", s):
        return "by"
    return None                                             # GFDL-only, FAL, unknown -> drop


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def commons(args, state, seen, hashes):
    total = 0
    for q in QUERIES + COMMONS_EXTRA:
        off = state["commons_offsets"].get(q, 0)
        got_q = 0
        while got_q < args.max_per_query and total < args.max_total and off < 2000:
            params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                      "gsrsearch": f"{q} filetype:bitmap", "gsrlimit": 50, "gsroffset": off,
                      "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 2048,
                      "iiextmetadatafilter": "License|LicenseShortName|UsageTerms|Artist|LicenseUrl"}
            j = json.loads(http("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
            pages = (j.get("query") or {}).get("pages", {})
            if not pages:
                break
            rows = []
            for p in pages.values():
                ii = (p.get("imageinfo") or [{}])[0]
                if ii.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(ii.get("width") or 0, ii.get("height") or 0) < args.min_side:
                    continue
                meta = ii.get("extmetadata", {})
                lic = commons_licence(meta)
                if lic is None or (lic == "by-sa" and not args.by_sa):
                    continue
                item = dict(source="commons", provider="wikimedia", id=str(p["pageid"]), query=q,
                            title=p.get("title", ""), creator=strip_html(meta.get("Artist", {}).get("value", ""))[:200],
                            creator_url="", license=lic, license_version="",
                            license_url=meta.get("LicenseUrl", {}).get("value", ""), share_alike=int(lic == "by-sa"),
                            landing_url=ii.get("descriptionurl", ""), image_url=ii.get("thumburl") or ii.get("url"),
                            width=ii.get("width"), height=ii.get("height"))
                if args.dry_run:
                    rows.append({**{k: item.get(k, "") for k in FIELDS}})
                    continue
                try:
                    row = store(item, seen, hashes, args)
                except Exception as e:
                    print(f"    skip {p['pageid']}: {e}", flush=True)
                    continue
                if row:
                    rows.append(row)
                time.sleep(0.2)
            if args.dry_run:
                write_candidates(rows)
            else:
                append_rows(rows)
            got_q += len(rows); total += len(rows)
            off = (j.get("continue") or {}).get("gsroffset")
            state["commons_offsets"][q] = off if off is not None else 10**6
            save_state(state)
            if off is None:
                break
            time.sleep(1.0)
        print(f"  commons {q!r}: +{got_q} (total {total})", flush=True)
    return total


def write_candidates(rows):
    fn = os.path.join(OUT, "candidates.csv")
    new = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-sa", action="store_true", help="also keep CC BY-SA (flagged share_alike=1)")
    ap.add_argument("--dry-run", action="store_true", help="metadata only, writes photos/candidates.csv")
    ap.add_argument("--max-per-query", type=int, default=120)
    ap.add_argument("--max-total", type=int, default=5000)
    ap.add_argument("--min-side", type=int, default=800, help="drop images smaller than this (px, short side)")
    ap.add_argument("--dup-bits", type=int, default=4, help="dHash Hamming distance treated as duplicate")
    ap.add_argument("--only", choices=["openverse", "commons"])
    ap.add_argument("--set", choices=["ground", "roofs"], default="ground",
                    help="query set; roofs -> photos_roofs/ (separate manifest and resume state)")
    args = ap.parse_args()
    if args.set == "roofs":
        global OUT, MANIFEST, STATE, QUERIES, COMMONS_EXTRA
        OUT = os.path.join(HERE, "photos_roofs")
        MANIFEST = os.path.join(OUT, "manifest.csv")
        STATE = os.path.join(OUT, "state.json")
        QUERIES, COMMONS_EXTRA = ROOF_QUERIES, ROOF_COMMONS_EXTRA
    if args.by_sa:
        ALLOWED.add("by-sa")
    os.makedirs(OUT, exist_ok=True)
    print(VERSION, "| downloader:", "curl" if USE_CURL else "urllib", "| Pillow:", HAVE_PIL, flush=True)
    if not HAVE_PIL:
        print("  (no Pillow: exact-duplicate check only, full-size files. `python3 -m pip install pillow` to enable)")
    state = load_state()
    seen, hashes = load_manifest()
    n = 0
    if args.only in (None, "commons"):
        n += commons(args, state, seen, hashes)            # no daily cap -> first
    if args.only in (None, "openverse"):
        n += openverse(args, state, seen, hashes)
    print(f"\ndone: {n} new {'candidates' if args.dry_run else 'images'}; manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
