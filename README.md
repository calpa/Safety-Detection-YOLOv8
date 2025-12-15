# Safety Detection YOLOv8 (Streamlit Demo)

Interactive PPE/safety detection demo built on **Ultralytics YOLOv8** with a **Streamlit UI**.

The app is designed for demos and debugging:

- Run detection on the included video (`videos/test.mp4`)
- Filter which classes to display
- Review a table of sampled frames with a red-flag dashboard (e.g. `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`)
- Cache outputs to disk so you don’t re-run inference on every UI change

## Demo UI

![Safety Detection YOLOv8](demo.png)

## Requirements

- Python 3.8+
- A YOLOv8 weights file: `ppe.pt` (place it in the project root)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

## How to use

1. Open **Demo: videos/test.mp4**.
2. Adjust:
   - Confidence threshold
   - IoU threshold
   - Class filter (which labels are drawn/returned)
3. Click **Run** under the original video.
4. Inspect:
   - The output table (frame index, timestamp, detection counts)
   - The red-flag dashboard totals
   - A specific frame’s annotated image + detections JSON

## Outputs (on-disk cache)

To keep the UI responsive, the app writes a cache under:

```text
outputs/streamlit_cache/<cache_id>/
```

Each cache entry contains:

```text
frames/frame_000024.jpg   # annotated frame image
json/frame_000024.json    # detections for that frame
rows.json                 # table summary
meta.json                 # settings + video metadata
```

The cache key includes the model path, thresholds, selected classes, and sampling parameters.

## Docker

Build the image:

```bash
docker build -t safety-detection-yolov8 .
```

Run the app:

```bash
docker run --rm -p 8501:8501 safety-detection-yolov8
```

### Providing the model weights (`ppe.pt`)

By default, the container expects `ppe.pt` to be available in the working directory (`/app/ppe.pt`).

If you keep `ppe.pt` on your host machine (recommended), mount it into the container:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/ppe.pt:/app/ppe.pt" \
  safety-detection-yolov8
```

### Mount videos and persist outputs (optional)

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/ppe.pt:/app/ppe.pt" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/outputs:/app/outputs" \
  safety-detection-yolov8
```

## Notes

- If you change thresholds/classes/sampling, you’ll generate a new cache entry.
- Use **Clear cache** to delete the current cache entry.

## License

MIT License. See [LICENSE](LICENSE).
