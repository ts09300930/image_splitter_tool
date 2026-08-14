import io
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(
    page_title="画像分割・高画質化",
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

ENHANCE_MODES = {
    "標準（おすすめ：2倍 + 自然なシャープ化）": "standard",
    "軽量（等倍 + 軽いシャープ化）": "light",
    "高画質化なし": "off",
}


def open_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    # 透過画像でもJPEG出力できるように、表示色を白背景に統一する
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

            # 白い区切り線などを少し削りたい時の安全な内側トリム
            tx0 = min(x0 + trim_px, x1 - 1)
            ty0 = min(y0 + trim_px, y1 - 1)
            tx1 = max(x1 - trim_px, tx0 + 1)
            ty1 = max(y1 - trim_px, ty0 + 1)

            pieces.append(image.crop((tx0, ty0, tx1, ty1)))

    return pieces


def enhance_image(image, mode):
    if mode == "off":
        return image

    if mode == "standard":
        # AIモデルを使わず、Community Cloudで安定しやすい高品質CPU処理。
        # 失われた情報を生成するものではないが、拡大時のギザつきと軽いボケを抑える。
        w, h = image.size
        image = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=1.2,
                percent=115,
                threshold=3,
            )
        )
        image = ImageEnhance.Sharpness(image).enhance(1.03)
        return image

    # 軽量モード：サイズは変えず、軽く輪郭を整える
    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=0.9,
            percent=90,
            threshold=3,
        )
    )
    return image


def image_to_bytes(image, fmt, jpeg_quality=95):
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
        mime = "image/jpeg"
        ext = "jpg"
    else:
        image.save(
            buffer,
            format="PNG",
            optimize=True,
            compress_level=6,
        )
        mime = "image/png"
        ext = "png"

    return buffer.getvalue(), mime, ext


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


st.title("🖼️ 画像分割・高画質化ツール")
st.caption(
    "画像を4・6・8・9・10・12・15分割し、必要なら自然に高画質化して、"
    "個別またはZIPでまとめて保存できます。外部AI APIは使いません。"
)

uploaded_file = st.file_uploader(
    "分割したい画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

left, right = st.columns([1, 1])

with left:
    preset_name = st.selectbox("分割レイアウト", list(PRESETS.keys()), index=2)

    if preset_name == "カスタム":
        c1, c2 = st.columns(2)
        with c1:
            cols = st.number_input("列数", min_value=1, max_value=10, value=4, step=1)
        with c2:
            rows = st.number_input("行数", min_value=1, max_value=10, value=2, step=1)
        cols, rows = int(cols), int(rows)
    else:
        cols, rows = PRESETS[preset_name]

    trim_px = st.slider(
        "各セルの端を内側へ削る量（白い区切り線の除去用）",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="区切り線が無い画像なら0でOKです。",
    )

with right:
    enhance_label = st.selectbox(
        "高画質化",
        list(ENHANCE_MODES.keys()),
        index=0,
    )
    enhance_mode = ENHANCE_MODES[enhance_label]

    output_format = st.radio(
        "出力形式",
        ["JPEG", "PNG"],
        horizontal=True,
        index=0,
    )

    jpeg_quality = 95
    if output_format == "JPEG":
        jpeg_quality = st.slider(
            "JPEG品質",
            min_value=85,
            max_value=100,
            value=95,
            step=1,
        )

if uploaded_file is not None:
    image = open_uploaded_image(uploaded_file)
    st.subheader("元画像")
    st.write(f"サイズ: **{image.width} × {image.height}px** / 分割数: **{cols * rows}枚**")
    st.image(image, caption=uploaded_file.name)

    if st.button("分割・高画質化する", type="primary", use_container_width=True):
        stem = Path(uploaded_file.name).stem
        pieces = split_image(image, cols=cols, rows=rows, trim_px=trim_px)

        files = []
        processed_previews = []

        progress = st.progress(0)
        total = len(pieces)

        for i, piece in enumerate(pieces, start=1):
            processed = enhance_image(piece, enhance_mode)
            data, mime, ext = image_to_bytes(
                processed,
                output_format,
                jpeg_quality=jpeg_quality,
            )
            filename = f"{stem}_{i:02d}.{ext}"
            files.append((filename, data, mime))
            processed_previews.append((filename, data, mime, processed.size))
            progress.progress(i / total)

        zip_data = build_zip(files)

        st.session_state["processed_previews"] = processed_previews
        st.session_state["zip_data"] = zip_data
        st.session_state["zip_name"] = f"{stem}_split_{len(files)}.zip"

        progress.empty()
        st.success(f"{len(files)}枚の処理が完了しました。")

if "processed_previews" in st.session_state:
    previews = st.session_state["processed_previews"]

    st.divider()
    st.subheader("分割結果")

    for start in range(0, len(previews), 4):
        row_items = previews[start:start + 4]
        columns = st.columns(len(row_items))

        for column, item in zip(columns, row_items):
            filename, data, mime, size = item
            with column:
                st.image(data, caption=f"{filename}\n{size[0]} × {size[1]}px")
                st.download_button(
                    label="この画像を保存",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"download_{filename}",
                    use_container_width=True,
                )

    st.download_button(
        label="📦 ZIPで全部まとめて保存",
        data=st.session_state["zip_data"],
        file_name=st.session_state["zip_name"],
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

st.divider()
st.caption(
    "「標準」モードはPillowのLANCZOS拡大と控えめなアンシャープ処理です。"
    "明るさやコントラストは意図的にほぼ触らず、元画像の雰囲気を変えにくい設定にしています。"
)
