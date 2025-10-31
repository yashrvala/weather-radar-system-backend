from flask import Flask, jsonify, send_file
from flask_cors import CORS
import requests, gzip, io, os, time, datetime
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)

OUTPUT_IMAGE = os.path.join(os.getcwd(), "radar.png")

# Possible MRMS paths (we'll try each until one works)
BASE_PATHS = [
    "https://noaa-mrms-pds.s3.amazonaws.com/MRMS/ReflectivityAtLowestAltitude/00.50",
    "https://noaa-mrms-pds.s3.amazonaws.com/MRMS/ReflectivityAtLowestAltitude",
    "https://noaa-mrms-pds.s3.amazonaws.com/MRMS/RadarOnly/CONUS/00.50",
]

def pick_reflectivity_var(ds):
    for v in ds.data_vars:
        if "reflect" in v.lower():
            return v
    return list(ds.data_vars)[0]


def get_bounds_from_dataset(ds):
    lat = lon = None
    for name in ("latitude", "lat", "Latitude", "LATITUDE"):
        if name in ds.coords:
            lat = ds.coords[name].values
            break
    for name in ("longitude", "lon", "Longitude", "LONGITUDE"):
        if name in ds.coords:
            lon = ds.coords[name].values
            break
    if lat is None or lon is None:
        return [25.0, -125.0, 50.0, -65.0]
    return [float(np.min(lat)), float(np.min(lon)), float(np.max(lat)), float(np.max(lon))]


def generate_image_from_grib(data_bytes):
    ds = xr.open_dataset(io.BytesIO(data_bytes), engine="cfgrib", backend_kwargs={"indexpath": ""})
    var_name = pick_reflectivity_var(ds)
    reflectivity = np.array(ds[var_name].values)
    reflectivity = np.ma.masked_invalid(reflectivity)
    bounds = get_bounds_from_dataset(ds)
    extent = (bounds[1], bounds[3], bounds[0], bounds[2])

    plt.figure(figsize=(10, 6), dpi=150)
    plt.imshow(reflectivity, origin="lower", extent=extent, cmap="turbo", vmin=-32, vmax=75)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_IMAGE, bbox_inches="tight", pad_inches=0)
    plt.close()
    return bounds


def try_fetch_latest_file():
    """Try to find the most recent available MRMS radar file."""
    now = datetime.datetime.utcnow()
    for minute_offset in range(0, 240, 2):  # try up to last 4 hours, every 2 mins
        dt = now - datetime.timedelta(minutes=minute_offset)
        fname = f"MRMS_ReflectivityAtLowestAltitude_00.50_{dt.strftime('%Y%m%d-%H%M00')}.grib2.gz"

        for base in BASE_PATHS:
            url = f"{base}/{fname}"
            r = requests.head(url)
            if r.status_code == 200:
                print(f"✅ Found file: {url}")
                return url
    return None


@app.route("/latest-meta", methods=["GET"])
def latest_meta():
    try:
        file_url = try_fetch_latest_file()
        if not file_url:
            return jsonify({"error": "No recent MRMS files found (checked multiple folders)"}), 500

        r = requests.get(file_url, timeout=60)
        if r.status_code != 200:
            return jsonify({"error": f"Failed to fetch GRIB2 (HTTP {r.status_code})"}), 500

        data = gzip.decompress(r.content)
        bounds = generate_image_from_grib(data)
        ts = int(time.time())

        return jsonify({"bounds": bounds, "timestamp": ts, "file": file_url})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/latest-image", methods=["GET"])
def latest_image():
    if not os.path.exists(OUTPUT_IMAGE):
        return jsonify({"error": "Image not yet generated. Call /latest-meta first."}), 400
    return send_file(OUTPUT_IMAGE, mimetype="image/png")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Weather Radar Backend Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
