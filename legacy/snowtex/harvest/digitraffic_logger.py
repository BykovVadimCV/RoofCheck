#!/usr/bin/env python3
"""
Winter logger for Finnish road-weather cameras (Fintraffic / digitraffic.fi, licence CC BY 4.0).

Each saved frame is paired with the readings of the camera's nearest road-weather station
taken at the same time: air and road temperature, road condition (dry / moist / wet / ice / snow ...),
snow, ice and water amount on the road surface in mm (optical sensor, where installed), friction,
precipitation type. That gives weak labels in millimetres for real fixed-camera winter frames.

    python3 digitraffic_logger.py                      # run forever: every 60 min, 60 cameras
    python3 digitraffic_logger.py --once               # one round, then exit (for cron / launchd)
    python3 digitraffic_logger.py --cameras 120 --interval 30
    python3 digitraffic_logger.py --all-weather        # also keep warm/dry frames (default: winter-ish only)
    python3 digitraffic_logger.py --backfill           # also pull the last 24 h of images (no sensor pairing)

Keep the Mac awake while it runs:  caffeinate -i python3 digitraffic_logger.py
Disk: ~150-270 KB per frame. 60 cameras x 24 frames/day ~ 250-400 MB per winter day.

Output: cams/<presetId>/<YYYYmmddTHHMMZ>.jpg, cams/frames.csv, cams/selection.json
Attribution required by the licence: "Source: Fintraffic / digitraffic.fi, license CC BY 4.0".
Faces/plates are rare at road cameras but frames still go through the blur step (M3) before use.
"""
import argparse, csv, hashlib, json, math, os, random, shutil, subprocess, sys, time, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cams")
FRAMES = os.path.join(OUT, "frames.csv")
SEL = os.path.join(OUT, "selection.json")
API = "https://tie.digitraffic.fi"
HDR = {"User-Agent": "IntelSet-snow-logger/1.0 (contact bykov2004@gmail.com)",
       "Digitraffic-User": "IntelSet/snow-clearance-research"}
LICENCE = "CC BY 4.0 - Source: Fintraffic / digitraffic.fi"
VERSION = "digitraffic_logger v1"
USE_CURL = sys.platform == "darwin" and shutil.which("curl") is not None

SENSORS = {  # station sensor name -> column
    "ILMA": "air_c", "TIE_1": "road_c", "KELI_1": "road_condition", "TIENPINNAN_TILA_1": "surface_state",
    "LUMEN_MÄÄRÄ1": "snow_mm", "JÄÄN_MÄÄRÄ1": "ice_mm", "VEDEN_MÄÄRÄ1": "water_mm", "KITKA1": "friction",
    "SATEEN_OLOMUOTO_PWDXX": "precip_type", "SADE_INTENSITEETTI": "precip_mm_h", "NÄKYVYYS_M": "visibility_m",
    "VALOISAA": "daylight",
}
COLS = ["preset", "camera", "camera_name", "lat", "lon", "image_time", "logged_at", "file", "bytes", "sha256",
        "weather_station", "sensor_time"] + list(dict.fromkeys(SENSORS.values())) + ["licence"]


def get(url):
    if USE_CURL:
        r = subprocess.run(["curl", "-fsSL", "--compressed", "--retry", "3"] +
                           sum([["-H", f"{k}: {v}"] for k, v in HDR.items()], []) + [url],
                           capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"curl {r.returncode}: {r.stderr.decode(errors='ignore')[:160]}")
        return r.stdout
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60) as r:
        return r.read()


def jget(path):
    return json.loads(get(API + path))


def weather_now():
    """{station_id: {column: value, '_time': iso}} for all road-weather stations, one request."""
    out = {}
    for st in jget("/api/weather/v1/stations/data").get("stations", []):
        row = {}
        for v in st.get("sensorValues", []):
            col = SENSORS.get(v.get("name"))
            if col:
                desc = v.get("sensorValueDescriptionEn")
                row[col] = f"{v.get('value')}" + (f" ({desc})" if desc and col in ("road_condition", "precip_type") else "")
                row["_time"] = max(row.get("_time", ""), v.get("measuredTime", ""))
        out[st["id"]] = row
    return out


def select_cameras(n, seed=7):
    """Prefer cameras whose nearest weather station has the optical snow/ice sensor; spread over latitude."""
    if os.path.exists(SEL):
        return json.load(open(SEL))
    feats = jget("/api/weathercam/v1/stations")["features"]
    wx = weather_now()
    cands = []
    for f in feats:
        p = f["properties"]
        if p.get("collectionStatus") != "GATHERING":
            continue
        presets = [q["id"] for q in p.get("presets", []) if q.get("inCollection")]
        if not presets:
            continue
        cands.append((p["id"], f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0], presets))
    print(f"  {len(cands)} active cameras; reading details ...", flush=True)
    random.Random(seed).shuffle(cands)
    detailed = []
    for cid, lat, lon, presets in cands[: max(4 * n, 200)]:
        try:
            d = jget(f"/api/weathercam/v1/stations/{cid}")["properties"]
        except Exception:
            continue
        ws = d.get("nearestWeatherStationId")
        has_snow = ws in wx and "snow_mm" in wx[ws]
        detailed.append(dict(camera=cid, name=(d.get("names") or {}).get("en") or d.get("name"), lat=lat, lon=lon,
                             weather_station=ws, snow_sensor=has_snow, presets=presets[:1]))   # first view only
        time.sleep(0.1)
    # snow-sensor cameras first, then stratify by latitude band so the north is represented
    detailed.sort(key=lambda c: (not c["snow_sensor"], random.random()))
    bands = {}
    for c in detailed:
        bands.setdefault(int(c["lat"]), []).append(c)
    pick = []
    while len(pick) < n and any(bands.values()):
        for b in sorted(bands):
            if bands[b] and len(pick) < n:
                pick.append(bands[b].pop(0))
    os.makedirs(OUT, exist_ok=True)
    json.dump(pick, open(SEL, "w"), indent=1, ensure_ascii=False)
    print(f"  selected {len(pick)} cameras ({sum(c['snow_sensor'] for c in pick)} with snow sensor) -> {SEL}")
    return pick


def wintery(w):
    def num(k):
        try:
            return float(str(w.get(k, "")).split()[0])
        except Exception:
            return None
    air, snow, ice = num("air_c"), num("snow_mm"), num("ice_mm")
    cond = str(w.get("road_condition", "")).lower()
    return (air is not None and air <= 3.0) or (snow or 0) > 0 or (ice or 0) > 0 or any(
        s in cond for s in ("ice", "snow", "frost", "slush"))


def append(rows):
    new = not os.path.exists(FRAMES)
    with open(FRAMES, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def save(cam, preset, img, t_img, w, ws):
    stamp = t_img.strftime("%Y%m%dT%H%MZ")
    fn = os.path.join(OUT, preset, f"{stamp}.jpg")
    if os.path.exists(fn):
        return None
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    open(fn, "wb").write(img)
    return {"preset": preset, "camera": cam["camera"], "camera_name": cam["name"], "lat": cam["lat"],
            "lon": cam["lon"], "image_time": t_img.isoformat(), "logged_at": datetime.now(timezone.utc).isoformat(),
            "file": os.path.relpath(fn, OUT), "bytes": len(img), "sha256": hashlib.sha256(img).hexdigest(),
            "weather_station": ws or "", "sensor_time": (w or {}).get("_time", ""),
            **{k: v for k, v in (w or {}).items() if k in COLS}, "licence": LICENCE}


def round_live(cams, args):
    wx = weather_now()
    rows, skipped = [], 0
    for cam in cams:
        w = wx.get(cam["weather_station"], {})
        if not args.all_weather and not wintery(w):
            skipped += 1
            continue
        for preset in cam["presets"]:
            try:
                img = get(f"https://weathercam.digitraffic.fi/{preset}.jpg")
            except Exception as e:
                print(f"  {preset}: {e}", flush=True)
                continue
            r = save(cam, preset, img, datetime.now(timezone.utc).replace(second=0, microsecond=0), w,
                     cam["weather_station"])
            if r:
                rows.append(r)
            time.sleep(0.2)
    append(rows)
    return len(rows), skipped


def backfill(cams):
    rows = []
    for cam in cams:
        try:
            h = jget(f"/api/weathercam/v1/stations/{cam['camera']}/history")
        except Exception as e:
            print(f"  history {cam['camera']}: {e}")
            continue
        for p in h.get("presets", []):
            for it in p.get("history", [])[::3]:                       # every 3rd = ~30 min
                url = it["imageUrl"]
                preset = url.rsplit("/", 1)[1].split(".")[0]
                if preset not in cam["presets"]:
                    continue
                t = datetime.fromisoformat(it["lastModified"].replace("Z", "+00:00"))
                try:
                    r = save(cam, preset, get(url), t, None, cam["weather_station"])
                except Exception:
                    continue
                if r:
                    rows.append(r)
                time.sleep(0.2)
    append(rows)
    print(f"  backfill: {len(rows)} frames (no sensor pairing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=60)
    ap.add_argument("--interval", type=int, default=60, help="minutes between rounds")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--all-weather", action="store_true", help="keep frames regardless of temperature/road state")
    ap.add_argument("--backfill", action="store_true", help="also fetch the last 24 h of images once")
    args = ap.parse_args()
    print(VERSION, "| downloader:", "curl" if USE_CURL else "urllib", flush=True)
    cams = select_cameras(args.cameras)
    if args.backfill:
        backfill(cams)
    while True:
        t0 = time.time()
        try:
            n, sk = round_live(cams, args)
            print(f"{datetime.now():%Y-%m-%d %H:%M} saved {n} frames, {sk} cameras skipped (not winter conditions)",
                  flush=True)
        except Exception as e:
            print(f"{datetime.now():%Y-%m-%d %H:%M} round failed: {e}", flush=True)
        if args.once:
            break
        time.sleep(max(60, args.interval * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
