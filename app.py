import io
import time
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="画像分割・人物高画質化",
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

MODES = {
    "人物・肌リアル（EDSR x2 + 質感復元 / おすすめ）": "skin_edsr",
    "人物・肌リアル（軽量 x2 / 高速）": "skin_lanczos",
    "EDSR x2のみ（比較用）": "edsr",
    "高画質化なし": "off",
}

EDSR_MODEL = {
    "name": "EDSR_x2.pb",
    "url": "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x2.pb",
    "algo": "edsr",
    "scale": 2,
    # Community Cloudのメモリを食い過ぎないよう小さめにタイル処理
    "tile_size": 220,
}


def open_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")

    return image.convert("RGB")


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

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 Streamlit-Skin-Enhancer"},
    )

    with urllib.request.urlopen(req, timeout=90) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    tmp.replace(dst)


def ensure_edsr_model():
    model_dir = Path("/tmp/streamlit_sr_models")
    model_path = model_dir / EDSR_MODEL["name"]

    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        download_file(EDSR_MODEL["url"], model_path)

    return model_path


@st.cache_resource(show_spinner=False)
def load_edsr():
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "cv2.dnn_superres が見つかりません。requirements.txt で "
            "opencv-contrib-python-headless を使用してください。"
        )

    model_path = ensure_edsr_model()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(EDSR_MODEL["algo"], EDSR_MODEL["scale"])
    return sr


def tiled_edsr(bgr: np.ndarray) -> np.ndarray:
    sr = load_edsr()
    scale = EDSR_MODEL["scale"]
    tile_size = EDSR_MODEL["tile_size"]

    h, w = bgr.shape[:2]

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


def make_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """
    肌色の「候補」を柔らかく推定するだけのマスク。
    顔認識はしないので、人物の顔立ちを作り直す処理は行わない。
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    # かなり広めに取って、境界をぼかす。
    lower = np.array([0, 128, 72], dtype=np.uint8)
    upper = np.array([255, 183, 142], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=7.0, sigmaY=7.0)

    mask_f = mask.astype(np.float32) / 255.0
    # 0/1に切りすぎないよう、最大でも0.9程度の効きにする
    return np.clip(mask_f * 0.9, 0.0, 0.9)


def restore_skin_texture(
    bgr: np.ndarray,
    texture_strength: int = 45,
    skin_smoothness: int = 14,
    local_contrast: int = 18,
) -> np.ndarray:
    """
    目的:
    - 肌をのっぺりさせない
    - 元画像に残っている微細な陰影を少し見えやすくする
    - 色ノイズだけ軽く抑える
    - 顔形状やパーツを生成し直さない

    「毛穴をAI生成する」処理ではない。
    元画像に残る高周波成分と局所コントラストを丁寧に持ち上げる。
    """
    bgr = bgr.copy()
    skin_mask = make_skin_mask(bgr)
    skin_mask_3 = skin_mask[..., None]

    # 1) 輝度と色を分離。
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)

    # 2) 色ムラだけをごく軽く整える。
    #    輝度Lにはデノイズをかけず、肌の細かな陰影を残す。
    smooth = max(0, min(30, skin_smoothness))
    if smooth > 0:
        sigma_color = 2.5 + smooth * 0.22
        a_s = cv2.bilateralFilter(a, d=5, sigmaColor=sigma_color, sigmaSpace=5)
        b_s = cv2.bilateralFilter(bb, d=5, sigmaColor=sigma_color, sigmaSpace=5)

        a_f = a.astype(np.float32)
        b_f = bb.astype(np.float32)
        a_s = a_s.astype(np.float32)
        b_s = b_s.astype(np.float32)

        chroma_mix = np.clip(smooth / 30.0, 0.0, 1.0) * skin_mask
        a = np.clip(a_f * (1 - chroma_mix) + a_s * chroma_mix, 0, 255).astype(np.uint8)
        bb = np.clip(b_f * (1 - chroma_mix) + b_s * chroma_mix, 0, 255).astype(np.uint8)

    # 3) 局所コントラストを弱く足す。
    #    CLAHEをそのまま100%使うと肌が硬くなるのでブレンド。
    lc = max(0, min(40, local_contrast))
    if lc > 0:
        clahe = cv2.createCLAHE(
            clipLimit=1.05 + lc * 0.012,
            tileGridSize=(8, 8),
        )
        l_clahe = clahe.apply(l)

        mix = (lc / 40.0) * 0.28
        l_f = l.astype(np.float32)
        l_c = l_clahe.astype(np.float32)

        # 肌は少し強め、背景はかなり弱め
        local_mask = 0.18 + 0.82 * skin_mask
        local_mix = mix * local_mask
        l = np.clip(
            l_f * (1 - local_mix) + l_c * local_mix,
            0,
            255,
        ).astype(np.uint8)

    # 4) 元々残っている微細ディテールを高周波成分として戻す。
    strength = max(0, min(100, texture_strength)) / 100.0
    if strength > 0:
        l_f = l.astype(np.float32)
        blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.15, sigmaY=1.15)
        detail = l_f - blur

        # 強いエッジのハローを抑えるため高周波を制限。
        detail = np.clip(detail, -10.0, 10.0)

        # 肌は最大0.62、背景は最大0.25程度。
        amount = strength * (0.24 + 0.38 * skin_mask)

        l = np.clip(l_f + detail * amount, 0, 255).astype(np.uint8)

    out_lab = cv2.merge([l, a, bb])
    out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    # 5) 元画像と軽くブレンドして「加工感」を抑える。
    #    肌候補部分だけ処理をやや強く反映。
    base = bgr.astype(np.float32)
    processed = out.astype(np.float32)

    final_mix = 0.38 + 0.52 * skin_mask_3
    out = np.clip(
        base * (1.0 - final_mix) + processed * final_mix,
        0,
        255,
    ).astype(np.uint8)

    return out


def upscale_lanczos(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return cv2.resize(
        bgr,
        (w * 2, h * 2),
        interpolation=cv2.INTER_LANCZOS4,
    )


def process_piece(
    image: Image.Image,
    mode: str,
    texture_strength: int,
    skin_smoothness: int,
    local_contrast: int,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mode == "off":
        out = bgr

    elif mode == "edsr":
        out = tiled_edsr(bgr)

    elif mode == "skin_edsr":
        out = tiled_edsr(bgr)
        out = restore_skin_texture(
            out,
            texture_strength=texture_strength,
            skin_smoothness=skin_smoothness,
            local_contrast=local_contrast,
        )

    elif mode == "skin_lanczos":
        out = upscale_lanczos(bgr)
        out = restore_skin_texture(
            out,
            texture_strength=texture_strength,
            skin_smoothness=skin_smoothness,
            local_contrast=local_contrast,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def image_to_bytes(image, fmt, jpeg_quality=97):
    buf = io.BytesIO()

    if fmt == "JPEG":
        image.convert("RGB").save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
            progressive=True,
        )
        return buf.getvalue(), "image/jpeg", "jpg"

    image.save(
        buf,
        format="PNG",
        optimize=True,
        compress_level=6,
    )
    return buf.getvalue(), "image/png", "png"


def build_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for filename, data, _mime in files:
            zf.writestr(filename, data)
    return buf.getvalue()


def clear_results():
    for key in (
        "processed_previews",
        "zip_data",
        "zip_name",
        "before_previews",
    ):
        st.session_state.pop(key, None)


st.title("🖼️ 画像分割・人物高画質化ツール v3")
st.caption(
    "今回は『解像度を上げる』より、人物写真の肌がのっぺりしないことを優先。"
    "顔を生成し直さず、元画像に残る微細な陰影・質感を見えやすくします。"
)

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    on_change=clear_results,
)

left, right = st.columns([1, 1])

with left:
    preset_name = st.selectbox(
        "分割レイアウト",
        list(PRESETS.keys()),
        index=2,
    )

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
        0, 20, 2, 1,
        help="区切り線が無ければ0。",
    )

with right:
    mode_label = st.selectbox(
        "処理モード",
        list(MODES.keys()),
        index=0,
    )
    mode = MODES[mode_label]

    output_format = st.radio(
        "出力形式",
        ["PNG", "JPEG"],
        horizontal=True,
        index=0,
        help="画質優先ならPNGがおすすめ。",
    )

    jpeg_quality = 97
    if output_format == "JPEG":
        jpeg_quality = st.slider("JPEG品質", 90, 100, 97, 1)

st.subheader("人物・肌の質感")
c1, c2, c3 = st.columns(3)

with c1:
    texture_strength = st.slider(
        "微細ディテール",
        0, 100, 45, 1,
        help="上げすぎると肌がザラつくので35〜55程度がおすすめ。",
    )

with c2:
    skin_smoothness = st.slider(
        "色ムラだけ軽く整える",
        0, 30, 12, 1,
        help="輝度の細かな凹凸は残し、色ノイズだけ弱く整えます。",
    )

with c3:
    local_contrast = st.slider(
        "肌の立体感",
        0, 40, 16, 1,
        help="局所コントラスト。上げすぎると加工感が出ます。",
    )

st.info(
    "最初は「微細ディテール45 / 色ムラ12 / 立体感16」がおすすめ。"
    "肌が硬く見えたら微細ディテールを35前後まで下げてください。"
)

if uploaded_file is not None:
    image = open_uploaded_image(uploaded_file)

    st.subheader("元画像")
    st.write(
        f"**{image.width} × {image.height}px** / "
        f"**{cols * rows}枚に分割**"
    )
    st.image(image, caption=uploaded_file.name)

    if st.button(
        "分割・人物高画質化する",
        type="primary",
        use_container_width=True,
    ):
        stem = Path(uploaded_file.name).stem
        pieces = split_image(image, cols, rows, trim_px)

        files = []
        previews = []
        before_previews = []

        status = st.empty()
        progress = st.progress(0)
        started = time.time()

        try:
            if mode in ("skin_edsr", "edsr"):
                status.info("EDSRモデルを準備しています…")
                load_edsr()

            for i, piece in enumerate(pieces, start=1):
                status.info(f"{i}/{len(pieces)} 枚目を処理中…")

                processed = process_piece(
                    piece,
                    mode=mode,
                    texture_strength=texture_strength,
                    skin_smoothness=skin_smoothness,
                    local_contrast=local_contrast,
                )

                data, mime, ext = image_to_bytes(
                    processed,
                    output_format,
                    jpeg_quality=jpeg_quality,
                )

                before_data, before_mime, _ = image_to_bytes(
                    piece,
                    "PNG",
                    jpeg_quality=100,
                )

                filename = f"{stem}_{i:02d}.{ext}"
                files.append((filename, data, mime))
                previews.append((filename, data, mime, processed.size))
                before_previews.append(
                    (f"元_{i:02d}", before_data, before_mime, piece.size)
                )

                progress.progress(i / len(pieces))

            st.session_state["processed_previews"] = previews
            st.session_state["before_previews"] = before_previews
            st.session_state["zip_data"] = build_zip(files)
            st.session_state["zip_name"] = (
                f"{stem}_split_skin_{len(files)}.zip"
            )

            elapsed = time.time() - started
            status.success(f"完了しました（約 {elapsed:.1f}秒）")

        except Exception as exc:
            status.error("処理中にエラーが発生しました。")
            st.exception(exc)
            st.caption(
                "Community CloudでEDSRが重い場合は、"
                "「人物・肌リアル（軽量 x2 / 高速）」に切り替えてください。"
            )

if (
    "processed_previews" in st.session_state
    and "before_previews" in st.session_state
):
    st.divider()
    st.subheader("元画像と処理後の比較")

    before = st.session_state["before_previews"]
    after = st.session_state["processed_previews"]

    # 最初の1枚を大きく比較
    b_name, b_data, _, b_size = before[0]
    a_name, a_data, _, a_size = after[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**分割直後**")
        st.image(b_data)
        st.caption(f"{b_size[0]} × {b_size[1]}px")
    with c2:
        st.markdown("**人物・肌高画質化後**")
        st.image(a_data)
        st.caption(f"{a_size[0]} × {a_size[1]}px")

    st.subheader("全画像")
    previews = st.session_state["processed_previews"]

    for start in range(0, len(previews), 4):
        row_items = previews[start:start + 4]
        columns = st.columns(len(row_items))

        for column, item in zip(columns, row_items):
            filename, data, mime, size = item
            with column:
                st.image(
                    data,
                    caption=f"{filename}\n{size[0]} × {size[1]}px",
                )
                st.download_button(
                    "この画像を保存",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"dl_{filename}",
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
    "このモードは顔パーツをAIで描き直すFace Restorationではありません。"
    "元画像に残る輝度の微細成分を持ち上げ、色ノイズだけ弱く整える方向です。"
)
