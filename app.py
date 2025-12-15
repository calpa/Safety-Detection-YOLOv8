import os
import tempfile
import json
import hashlib
import shutil

import cv2
import numpy as np
import streamlit as st

from safety_detection import classNames, detect_and_annotate, load_model


# Decode an uploaded image into an OpenCV BGR array.
def _bgr_from_uploaded_image(uploaded_file):
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img_bgr


# Process a full video into an annotated mp4 (used only by the Upload -> Video tab).
def _process_video_to_tempfile(input_path: str, model, conf: float, iou: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = st.progress(0)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        annotated, _ = detect_and_annotate(frame, model, conf_threshold=conf, iou_threshold=iou)
        writer.write(annotated)

        idx += 1
        if total_frames > 0:
            progress.progress(min(1.0, idx / total_frames))

    cap.release()
    writer.release()
    progress.empty()
    return out_path


def _get_video_meta(path: str):
    # Read basic metadata once (fps, frame count, dimensions) for UI controls.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }


def _read_frame_at(path: str, frame_index: int):
    # Random-access a specific frame (used for sampled processing + cache generation).
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _cache_key(payload) -> str:
    # Stable cache ID based on settings + sampling parameters.
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _ensure_empty_dir(path: str):
    # Ensure a clean output directory for a fresh run.
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _read_json(path: str):
    # Convenience helpers for persisted cache artifacts.
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    st.set_page_config(page_title="Safety Detection YOLOv8", layout="wide")
    st.title("Safety Detection YOLOv8 (Streamlit Demo)")

    st.markdown(
        """
        <style>
        .block-container { max-width: 95rem; }
        div[data-testid="stDataFrame"] { width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input("Weights path", value="ppe.pt")
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.68, 0.01)
        iou = st.slider("IoU threshold", 0.0, 1.0, 0.7, 0.01)
        selected_classes = st.multiselect(
            "Show labels",
            options=classNames,
            default=classNames,
        )

    @st.cache_resource
    def _get_model(path: str):
        # Cache the YOLO model in memory across Streamlit reruns.
        return load_model(path)

    if not os.path.exists(weights_path):
        st.error(f"Weights file not found: {weights_path}")
        st.stop()

    model = _get_model(weights_path)

    tab_demo, tab_upload = st.tabs(["Demo: videos/test.mp4", "Upload"])

    with tab_demo:
        # Demo mode uses the repo's default video and an on-disk cache for results.
        default_video_path = os.path.join("videos", "test.mp4")
        if not os.path.exists(default_video_path):
            st.error(f"Default video not found: {default_video_path}")
            st.stop()

        meta = _get_video_meta(default_video_path)
        if meta is None:
            st.error("Failed to open default video")
            st.stop()

        st.subheader("Original video")
        with open(default_video_path, "rb") as f:
            st.video(f.read())

        st.markdown("---")
        # Run triggers cache generation. Clear removes only this cache entry.
        run_clicked = st.button("Run", type="primary")
        clear_clicked = st.button("Clear cache")

        st.markdown("---")
        st.subheader("Output table + red-flag dashboard")

        total_frames = int(meta.get("total_frames", 0) or 0)
        fps = float(meta.get("fps", 25.0) or 25.0)

        if total_frames <= 0:
            start_frame = st.number_input("Start frame", min_value=0, value=0, step=1)
        else:
            start_frame = st.slider("Start frame", 0, max(0, total_frames - 1), 0, 1)

        if total_frames <= 0:
            end_frame = st.number_input("End frame", min_value=int(start_frame), value=int(start_frame), step=1)
        else:
            end_frame = st.slider("End frame", int(start_frame), max(0, total_frames - 1), max(0, total_frames - 1), 1)

        step = st.number_input("Step (frames)", min_value=1, max_value=500, value=10, step=1)

        allowed = set(selected_classes) if selected_classes else None
        red_flags = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

        output_root = os.path.join(os.getcwd(), "outputs", "streamlit_cache")
        os.makedirs(output_root, exist_ok=True)

            # Cache layout:
            # outputs/streamlit_cache/<cache_id>/
            #   frames/frame_000024.jpg
            #   json/frame_000024.json
            #   rows.json   (table summary)
            #   meta.json   (settings + video meta)

        cache_payload = {
            "video": default_video_path,
            "weights": weights_path,
            "conf": float(conf),
            "iou": float(iou),
            "classes": sorted(list(allowed)) if allowed is not None else None,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "step": int(step),
        }
        cache_id = _cache_key(cache_payload)
        cache_dir = os.path.join(output_root, cache_id)
        frames_dir = os.path.join(cache_dir, "frames")
        json_dir = os.path.join(cache_dir, "json")
        rows_path = os.path.join(cache_dir, "rows.json")
        meta_path = os.path.join(cache_dir, "meta.json")

        if clear_clicked:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            st.rerun()

        cache_exists = os.path.exists(rows_path) and os.path.exists(meta_path)
        if cache_exists:
            st.caption(f"Cache: hit ({cache_id})")
        else:
            st.caption(f"Cache: miss ({cache_id})")

        rows = []

        if run_clicked or not cache_exists:
            # Generate cached artifacts (sampled annotated frames + per-frame detections + summary table).
            _ensure_empty_dir(cache_dir)
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            effective_end = int(end_frame)
            if total_frames > 0:
                effective_end = min(effective_end, max(0, total_frames - 1))
            if effective_end < int(start_frame):
                effective_end = int(start_frame)

            frame_indices = list(range(int(start_frame), int(effective_end) + 1, int(step)))
            progress = st.progress(0)

            for idx, frame_index in enumerate(frame_indices):
                frame = _read_frame_at(default_video_path, int(frame_index))
                if frame is None:
                    break

                annotated, detections = detect_and_annotate(
                    frame.copy(),
                    model,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    allowed_class_names=allowed,
                )

                ts_seconds = float(frame_index) / fps if fps > 0 else 0.0
                red_counts = {k: 0 for k in red_flags}
                for d in detections:
                    name = d.get("class_name")
                    if name in red_counts:
                        red_counts[name] += 1
                has_red = sum(red_counts.values()) > 0

                frame_basename = f"frame_{int(frame_index):06d}"
                frame_img_path = os.path.join(frames_dir, f"{frame_basename}.jpg")
                frame_json_path = os.path.join(json_dir, f"{frame_basename}.json")

                cv2.imwrite(frame_img_path, annotated)
                _write_json(
                    frame_json_path,
                    {
                        "frame": int(frame_index),
                        "t_seconds": ts_seconds,
                        "detections": detections,
                    },
                )

                rows.append(
                    {
                        "frame": int(frame_index),
                        "t(s)": round(ts_seconds, 2),
                        "detections": int(len(detections)),
                        "NO-Hardhat": int(red_counts["NO-Hardhat"]),
                        "NO-Mask": int(red_counts["NO-Mask"]),
                        "NO-Safety Vest": int(red_counts["NO-Safety Vest"]),
                        "red_flag": bool(has_red),
                    }
                )

                if len(frame_indices) > 0:
                    progress.progress(min(1.0, float(idx + 1) / float(len(frame_indices))))

            progress.empty()

            _write_json(rows_path, rows)
            _write_json(
                meta_path,
                {
                    "cache_payload": cache_payload,
                    "video_meta": meta,
                    "frame_indices": [r["frame"] for r in rows],
                },
            )
            st.rerun()

        rows = _read_json(rows_path) or []

        red_frames = sum(1 for r in rows if r.get("red_flag"))
        total_no_hardhat = sum(int(r.get("NO-Hardhat", 0)) for r in rows)
        total_no_mask = sum(int(r.get("NO-Mask", 0)) for r in rows)
        total_no_vest = sum(int(r.get("NO-Safety Vest", 0)) for r in rows)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Processed frames", str(len(rows)))
        m2.metric("Frames with red flag", str(red_frames))
        m3.metric("NO-Hardhat total", str(total_no_hardhat))
        m4.metric("NO-Mask / NO-Vest", f"{total_no_mask} / {total_no_vest}")

        try:
            st.dataframe(rows, height=520, use_container_width=True)
        except TypeError:
            st.dataframe(rows, height=520)

        if len(rows) > 0:
            frame_choices = sorted([int(r["frame"]) for r in rows])

            def _nearest_frame(target: int) -> int:
                return min(frame_choices, key=lambda x: abs(x - target))

            default_frame = int(st.session_state.get("inspect_frame", frame_choices[0]))
            default_frame = _nearest_frame(default_frame)

            if "inspect_frame_slider" not in st.session_state:
                st.session_state.inspect_frame_slider = default_frame
            if "inspect_frame_input" not in st.session_state:
                st.session_state.inspect_frame_input = default_frame
            if "inspect_frame" not in st.session_state:
                st.session_state.inspect_frame = default_frame

            def _on_slider_change():
                value = int(st.session_state.inspect_frame_slider)
                value = _nearest_frame(value)
                st.session_state.inspect_frame = value
                st.session_state.inspect_frame_input = value

            def _on_input_change():
                value = int(st.session_state.inspect_frame_input)
                value = _nearest_frame(value)
                st.session_state.inspect_frame = value
                st.session_state.inspect_frame_slider = value

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.slider(
                    "Inspect frame",
                    min_value=frame_choices[0],
                    max_value=frame_choices[-1],
                    value=int(st.session_state.inspect_frame_slider),
                    step=max(1, int(step)),
                    key="inspect_frame_slider",
                    on_change=_on_slider_change,
                )
            with col_b:
                st.number_input(
                    "Go to",
                    min_value=int(frame_choices[0]),
                    max_value=int(frame_choices[-1]),
                    value=int(st.session_state.inspect_frame_input),
                    step=max(1, int(step)),
                    key="inspect_frame_input",
                    on_change=_on_input_change,
                )

            selected_frame = int(_nearest_frame(int(st.session_state.inspect_frame)))

            frame_basename = f"frame_{int(selected_frame):06d}"
            frame_img_path = os.path.join(frames_dir, f"{frame_basename}.jpg")
            frame_json_path = os.path.join(json_dir, f"{frame_basename}.json")

            annotated_bgr = cv2.imread(frame_img_path) if os.path.exists(frame_img_path) else None
            frame_json = _read_json(frame_json_path) or {}
            detections = frame_json.get("detections", [])

            if annotated_bgr is not None:
                st.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True,
                )
            st.subheader("Detections (selected frame)")
            st.json(detections)

    with tab_upload:
        tab_image, tab_video = st.tabs(["Image", "Video"])

        with tab_image:
            uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded_img is not None:
                img_bgr = _bgr_from_uploaded_image(uploaded_img)
                if img_bgr is None:
                    st.error("Failed to decode image")
                    st.stop()

                allowed = set(selected_classes) if selected_classes else None
                annotated, detections = detect_and_annotate(
                    img_bgr.copy(),
                    model,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    allowed_class_names=allowed,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                with col2:
                    st.subheader("Annotated")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                st.subheader("Detections (filtered)")
                st.json(detections)

        with tab_video:
            uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
            if uploaded_vid is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1]) as tmp:
                    tmp.write(uploaded_vid.read())
                    tmp_path = tmp.name

                st.info("Processing video. This may take a while depending on length.")
                out_path = _process_video_to_tempfile(tmp_path, model, conf=conf, iou=iou)

                with open(out_path, "rb") as f:
                    st.video(f.read())


if __name__ == "__main__":
    main()
