import cv2
import numpy as np

W, H = 640, 480
WINDOW_NAME = "SMCG Viewer"

# ----------------- ユーティリティ -----------------

def normalize(img, eps=1e-6):
    img = img.astype(np.float32)
    m = img.max()
    if m < eps:
        return np.zeros_like(img, dtype=np.float32)
    return img / m

# ボタン情報を保存する
buttons = []
current_mode = "COMPOSITE"  # COMPOSITE / STILL / MOTION / COH / GEN

def mouse_callback(event, x, y, flags, param):
    global current_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        for (name, (x1, y1, x2, y2)) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                current_mode = name
                print("mode:", name)
                break

def draw_buttons(canvas):
    """
    canvas: BGR画像(float32 0-1)
    上部にボタンを描画する
    """
    global buttons
    h, w, _ = canvas.shape
    bar_h = 40
    btn_w = 110
    margin = 5

    names = ["COMPOSITE", "STILL", "MOTION", "COH", "GEN"]

    buttons = []
    x = margin
    y1 = 0
    y2 = bar_h

    for name in names:
        x1 = x
        x2 = x + btn_w

        if name == current_mode:
            color = (0.9, 0.9, 0.2)  # active
            text_color = (0.1, 0.1, 0.1)
        else:
            color = (0.3, 0.3, 0.3)
            text_color = (0.9, 0.9, 0.9)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (color[2], color[1], color[0]),  # BGR
            thickness=-1
        )

        cv2.putText(
            canvas,
            name,
            (x1 + 8, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (text_color[2], text_color[1], text_color[0]),
            1,
            cv2.LINE_AA
        )

        buttons.append((name, (x1, y1, x2, y2)))
        x += btn_w + margin

    cv2.rectangle(
        canvas,
        (0, bar_h),
        (w, bar_h + 2),
        (0.0, 0.0, 0.0),
        thickness=-1
    )

def main():
    global current_mode

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    prev_gray = None
    motion_mean = None
    alpha_coh = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray_blur
            motion_mean = np.zeros_like(gray_blur, dtype=np.float32)
            continue

        # ---- Motion ----
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray_blur,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_raw = mag
        motion_norm = normalize(motion_raw)

        # ---- Stillness ----
        stillness = 1.0 - motion_norm
        stillness = cv2.GaussianBlur(stillness, (9, 9), 0)

        # ---- Coherence ----
        motion_mean = alpha_coh * motion_mean + (1 - alpha_coh) * motion_norm
        coh_raw = 1.0 - np.abs(motion_norm - motion_mean)
        coherence = normalize(coh_raw)
        coherence = cv2.GaussianBlur(coherence, (9, 9), 0)

        # ---- Genesys ----
        diff = motion_norm - motion_mean
        diff = np.clip(diff, 0, 1)
        genesys = normalize(diff)
        genesys = cv2.GaussianBlur(genesys, (5, 5), 0)

        # ---- 4つのマップ（0〜1）----
        s = np.clip(stillness, 0, 1)
        m = np.clip(motion_norm, 0, 1)
        c = np.clip(coherence, 0, 1)
        g = np.clip(genesys, 0, 1)

        # 元映像（0-1）
        frame_f = frame.astype(np.float32) / 255.0

        # ---- 各レイヤーの個別カラー（単体表示用）----
        base = frame_f * 0.2

        still_rgb = np.zeros_like(base)
        still_rgb[..., 0] = s * 0.8
        still_rgb[..., 1] = s * 0.1
        still_rgb[..., 2] = s * 0.2

        mot_rgb = np.zeros_like(base)
        mot_rgb[..., 2] = m * 1.0
        mot_rgb[..., 1] = m * 0.4

        coh_rgb = np.zeros_like(base)
        coh_rgb[..., 1] = c * 0.8
        coh_rgb[..., 0] = c * 0.8

        gen_rgb = np.zeros_like(base)
        gen_rgb[..., 2] = g * 0.9
        gen_rgb[..., 0] = g * 0.6

        # ---- 表示モードごとの画像を選択 ----
        if current_mode == "COMPOSITE":
            # ベースは元映像
            img = frame_f.copy()

            # 3chに拡張したマップ
            s3 = s[..., None]
            m3 = m[..., None]
            c3 = c[..., None]
            g3 = g[..., None]

            # 各レイヤーの色（0〜1）
            color_s = np.array([0.2, 0.6, 1.0], dtype=np.float32)  # 青
            color_m = np.array([0.1, 0.2, 1.0], dtype=np.float32)  # 赤
            color_c = np.array([0.7, 1.0, 0.9], dtype=np.float32)  # シアン
            color_g = np.array([0.9, 0.6, 1.0], dtype=np.float32)  # 紫

            # 全体のオーバーレイ強さ
            alpha = 0.35  # 0.0〜0.5くらいで好みに調整

            # 各マップの影響度（動きが強すぎて真っ白にならないように少し丸める）
            s_level = np.sqrt(s3)
            m_level = np.sqrt(m3)
            c_level = np.sqrt(c3)
            g_level = np.sqrt(g3)

            overlay = (
                s_level * color_s +
                m_level * color_m +
                c_level * color_c +
                g_level * color_g
            )

            # 元映像と線形補間（絶対 0〜1 の範囲内）
            img = (1.0 - alpha) * img + alpha * overlay

        elif current_mode == "STILL":
            img = still_rgb
        elif current_mode == "MOTION":
            img = mot_rgb
        elif current_mode == "COH":
            img = coh_rgb
        elif current_mode == "GEN":
            img = gen_rgb
        else:
            img = frame_f

        img = np.clip(img, 0, 1)

        # ボタン用のキャンバス
        canvas = np.zeros((H + 50, W, 3), dtype=np.float32)
        canvas[50:50 + H, :, :] = img
        draw_buttons(canvas)

        canvas_bgr = (canvas * 255).astype(np.uint8)
        cv2.imshow(WINDOW_NAME, canvas_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        prev_gray = gray_blur

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
