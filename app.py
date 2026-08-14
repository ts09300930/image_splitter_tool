import io
import os
import time
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(
    page_title="画像分割・AI高画質化",
    page_icon="🖼️",
    layout="wide",
)

PRESETS = {
    "4分割（2列 × 2行）": (2, 2),
    "6分割（3列 × 2行）": (3, 2),
    "8分割（4列 × 2行）": (4, 2),
    "9分割（3列 × 3行）": (3, 3),
    "10分割（5列 × 2行）": (5, 2),
    "12分割（4列 × 3行）": (4, 3),
    "15分割（5列 × 3行）": (5, 3),
    "カスタム": None,
}

QUALITY_MODES = {
    "AI軽量・安定（FSRCNN x2 / おすすめ）": "fsrcnn",
    "AI高品質（EDSR x2 / 遅め）": "edsr",
    "従来方式（LANCZOS x2）": "lanczos",
    "高画質化なし": "off",
}

MODEL_INFO = {
    "fsrcnn": {
        "name": "FSRCNN_x2.pb",
        "url": "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb",
        "algo": "fsrcnn",
        "scale": 2,
        "tile_size": 520,
    },
    "edsr": {
        "name": "EDSR_x2.pb",
        "url": "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x2.pb",
        "algo": "edsr",
        "scale": 2,
        "tile_size": 220,
    },
}


def open_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")

    return image


def split_image(image, cols, rows, trim_px=0):
    width, height = image.size
    pieces = []

    for row in range(rows):
        y0 = round(row * height / rows)
        y1 = round((row + 1) * height / rows)

        for col in range(cols):
            x0 = round(col * width / cols)
            x1 = round((col + 1) * width / cols)

            tx0 = min(x0 + trim_px, x1 - 1)
            ty0 = min(y0 + trim_px, y1 - 1)
            tx1 = max(x1 - trim_px, tx0 + 1)
            ty1 = max(y1 - trim_px, ty0 + 1)

            pieces.append(image.crop((tx0, ty0, tx1, ty1)))

    return pieces


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 Streamlit-Image-SR"},
    )

    with urllib.request.urlopen(request, timeout=90) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    tmp.replace(dst)


def ensure_model(mode: str) -> Path:
    info = MODEL_INFO[mode]
    model_dir = Path("/tmp/streamlit_sr_models")
    model_path = model_dir / info["name"]

    if not model_path.exists() or model_path.stat().st_size < 1000:
        download_file(info["url"], model_path)

    return model_path


@st.cache_resource(show_spinner=False)
def load_sr_model(mode: str):
    info = MODEL_INFO[mode]
    model_path = ensure_model(mode)

    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "cv2.dnn_superres が見つかりません。requirements.txt で "
            "opencv-contrib-python-headless を使用してください。"
        )

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(info["algo"], info["scale"])
    return sr


def tiled_super_resolution(bgr: np.ndarray, mode: str) -> np.ndarray:
    info = MODEL_INFO[mode]
    sr = load_sr_model(mode)
    scale = info["scale"]
    tile_size = info["tile_size"]

    h, w = bgr.shape[:2]

    # 小さめの画像はそのまま推論。大きい画像だけ分割してメモリ節約。
    if h <= tile_size and w <= tile_size:
        return sr.upsample(bgr)

    overlap = 18
    output = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    for y0 in range(0, h, tile_size):
        y1 = min(y0 + tile_size, h)

        for x0 in range(0, w, tile_size):
            x1 = min(x0 + tile_size, w)

            ex0 = max(0, x0 - overlap)
            ey0 = max(0, y0 - overlap)
            ex1 = min(w, x1 + overlap)
            ey1 = min(h, y1 + overlap)

            tile = bgr[ey0:ey1, ex0:ex1]
            up = sr.upsample(tile)

            # 重なり部分を捨てて、中央の本来のセルだけ貼る
            sx0 = (x0 - ex0) * scale
            sy0 = (y0 - ey0) * scale
            sx1 = sx0 + (x1 - x0) * scale
            sy1 = sy0 + (y1 - y0) * scale

            dx0 = x0 * scale
            dy0 = y0 * scale
            dx1 = x1 * scale
            dy1 = y1 * scale

            output[dy0:dy1, dx0:dx1] = up[sy0:sy1, sx0:sx1]

    return output


def subtle_finish(pil_image: Image.Image, strength: float = 0.18) -> Image.Image:
    """
    AI超解像後の輪郭だけほんの少し整える。
    強いシャープ化は肌・髪・輪郭が不自然になるので避ける。
    """
    sharpened = pil_image.filter(
        ImageFilter.UnsharpMask(radius=0.8, percent=65, threshold=4)
    )
    return Image.blend(pil_image, sharpened, strength)


def enhance_image(image: Image.Image, mode: str, finish_sharpen: bool) -> Image.Image:
    if mode == "off":
        return image

    if mode == "lanczos":
        w, h = image.size
        result = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        if finish_sharpen:
            result = subtle_finish(result, strength=0.28)
        return result

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    upscaled_bgr = tiled_super_resolution(bgr, mode)
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(upscaled_rgb)

    if finish_sharpen:
        result = subtle_finish(result)

    return result


def image_to_bytes(image, fmt, jpeg_quality=97):
    buffer = io.BytesIO()

    if fmt == "JPEG":
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
            progressive=True,
        )
        return buffer.getvalue(), "image/jpeg", "jpg"

    image.save(
        buffer,
        format="PNG",
        optimize=True,
        compress_level=6,
    )
    return buffer.getvalue(), "image/png", "png"


def build_zip(files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for filename, data, _mime in files:
            zf.writestr(filename, data)
    return zip_buffer.getvalue()


def clear_previous_results():
    for key in ("processed_previews", "zip_data", "zip_name", "last_settings"):
        st.session_state.pop(key, None)


st.title("🖼️ 画像分割・AI高画質化ツール v2")
st.caption(
    "分割はそのまま、画質改善だけを本物のニューラル超解像に変更した版です。"
    "外部の有料APIは使わず、Streamlit CloudのCPU上で処理します。"
)

uploaded_file = st.file_uploader(
    "分割したい画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    on_change=clear_previous_results,
)

left, right = st.columns([1, 1])

with left:
    preset_name = st.selectbox("分割レイアウト", list(PRESETS.keys()), index=2)

    if preset_name == "カスタム":
        c1, c2 = st.columns(2)
        with c1:
            cols = int(st.number_input("列数", 1, 10, 4, 1))
        with c2:
            rows = int(st.number_input("行数", 1, 10, 2, 1))
    else:
        cols, rows = PRESETS[preset_name]

    trim_px = st.slider(
        "区切り線を削る量",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="白い区切り線が無い画像なら0でOKです。",
    )

with right:
    quality_label = st.selectbox(
        "画質改善",
        list(QUALITY_MODES.keys()),
        index=0,
    )
    quality_mode = QUALITY_MODES[quality_label]

    finish_sharpen = st.checkbox(
        "最後にごく弱いシャープ処理",
        value=True,
        help="強くしすぎない設定です。顔や肌がカリカリになるのを避けます。",
    )

    output_format = st.radio("出力形式", ["JPEG", "PNG"], horizontal=True)
    jpeg_quality = 97
    if output_format == "JPEG":
        jpeg_quality = st.slider("JPEG品質", 90, 100, 97, 1)

if quality_mode == "fsrcnn":
    st.info(
        "おすすめ設定：FSRCNN x2。モデルが非常に軽く、無料Streamlit Cloudで動かしやすい構成です。"
    )
elif quality_mode == "edsr":
    st.warning(
        "EDSR x2はFSRCNNより重いです。初回のみ約37MBのモデル取得があり、"
        "CPU処理なので8枚では時間がかかることがあります。"
    )

if uploaded_file is not None:
    image = open_uploaded_image(uploaded_file)

    st.subheader("元画像")
    st.write(
        f"サイズ: **{image.width} × {image.height}px** / "
        f"分割数: **{cols * rows}枚**"
    )
    st.image(image, caption=uploaded_file.name)

    if st.button("分割・AI高画質化する", type="primary", use_container_width=True):
        stem = Path(uploaded_file.name).stem
        pieces = split_image(image, cols=cols, rows=rows, trim_px=trim_px)

        files = []
        previews = []

        status = st.empty()
        progress = st.progress(0)
        started = time.time()

        try:
            if quality_mode in ("fsrcnn", "edsr"):
                status.info("AI超解像モデルを準備しています…")
                load_sr_model(quality_mode)

            for i, piece in enumerate(pieces, start=1):
                status.info(f"{i}/{len(pieces)} 枚目を処理中…")
                processed = enhance_image(
                    piece,
                    quality_mode,
                    finish_sharpen=finish_sharpen,
                )

                data, mime, ext = image_to_bytes(
                    processed,
                    output_format,
                    jpeg_quality=jpeg_quality,
                )

                filename = f"{stem}_{i:02d}.{ext}"
                files.append((filename, data, mime))
                previews.append((filename, data, mime, processed.size))

                progress.progress(i / len(pieces))

            zip_data = build_zip(files)

            st.session_state["processed_previews"] = previews
            st.session_state["zip_data"] = zip_data
            st.session_state["zip_name"] = f"{stem}_split_ai_{len(files)}.zip"

            elapsed = time.time() - started
            status.success(f"完了しました（約 {elapsed:.1f} 秒）")

        except Exception as exc:
            status.error("AI高画質化でエラーが発生しました。")
            st.exception(exc)
            st.caption(
                "もし無料Cloud側でAI処理が厳しい場合は、"
                "「AI軽量・安定（FSRCNN x2）」を選んでください。"
            )

if "processed_previews" in st.session_state:
    st.divider()
    st.subheader("分割結果")

    previews = st.session_state["processed_previews"]

    for start in range(0, len(previews), 4):
        row_items = previews[start:start + 4]
        columns = st.columns(len(row_items))

        for column, item in zip(columns, row_items):
            filename, data, mime, size = item
            with column:
                st.image(data, caption=f"{filename}\n{size[0]} × {size[1]}px")
                st.download_button(
                    "この画像を保存",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"download_{filename}",
                    use_container_width=True,
                )

    st.download_button(
        "📦 ZIPで全部まとめて保存",
        data=st.session_state["zip_data"],
        file_name=st.session_state["zip_name"],
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

st.divider()
st.caption(
    "AI軽量＝FSRCNN x2 / AI高品質＝EDSR x2。"
    "どちらも画像全体を生成し直すGAN系ではなく、超解像向けのCNNモデルを使います。"
)
