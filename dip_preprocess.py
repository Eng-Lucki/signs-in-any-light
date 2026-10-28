"""Classical DIP stage for the ASL translator.

Three groups of functions, all operating on uint8 BGR images (OpenCV convention):

1. Degradations   - simulate bad capture conditions (dark room, sensor noise,
                    motion blur, backlight, heavy compression) so we can test
                    the recognizer under controlled, repeatable conditions.
2. Enhancements   - classical point / neighbourhood operations that try to undo
                    those problems (gamma, histogram equalization, CLAHE,
                    Gaussian / median / bilateral / non-local-means denoising,
                    unsharp masking) plus an adaptive pipeline that measures the
                    frame and only applies what it needs.
3. Quality metrics - no-reference measurements (brightness, contrast, noise,
                    sharpness) used by the adaptive pipeline, and full-reference
                    PSNR / SSIM used in the notebook.
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _lut(table):
    return np.clip(table, 0, 255).astype(np.uint8)


def _to_float(img):
    return img.astype(np.float32) / 255.0


def _to_uint8(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Degradations
# ---------------------------------------------------------------------------

def low_light(img, gamma=2.5, gain=0.6):
    """Dark exposure: out = gain * in^gamma (in [0, 1]). Crushes shadows first."""
    table = gain * (np.arange(256) / 255.0) ** gamma * 255.0
    return cv2.LUT(img, _lut(table))


def gaussian_noise(img, sigma=20, seed=0):
    """Additive zero-mean Gaussian noise with std `sigma` grey levels (sensor read noise)."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def low_light_noisy(img, gamma=2.5, sigma=12, seed=0):
    """Realistic dim room: dark exposure plus the sensor noise that comes with it."""
    return gaussian_noise(low_light(img, gamma), sigma, seed)


def motion_blur(img, length=15, angle=0):
    """Linear motion blur: convolve with a `length`-pixel line kernel at `angle` degrees."""
    length = max(1, int(length))
    if length == 1:
        return img.copy()
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def backlight(img, strength=0.7):
    """Bright window behind the signer: subject underexposed, glare from the top."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - w / 2) / w) ** 2 + (yy / h) ** 2
    glare = np.exp(-d2 / (2 * 0.35 ** 2))[..., None]
    subject = _to_float(img) * (1.0 - 0.65 * strength)
    out = subject + strength * glare
    return _to_uint8(out)


def jpeg(img, quality=10):
    """Heavy JPEG compression (8x8 block artifacts), e.g. a low-bandwidth video call."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# name -> (function, default kwargs, (sweep parameter, sweep values))
DEGRADATIONS = {
    "clean":           (lambda img, **k: img.copy(), {}, None),
    "low_light":       (low_light, {"gamma": 3.0}, ("gamma", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])),
    "low_light_noisy": (low_light_noisy, {"gamma": 2.5, "sigma": 12}, ("gamma", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])),
    "gaussian_noise":  (gaussian_noise, {"sigma": 35}, ("sigma", [0, 10, 20, 30, 40, 50, 60])),
    "motion_blur":     (motion_blur, {"length": 21}, ("length", [1, 5, 10, 15, 20, 30, 40])),
    "backlight":       (backlight, {"strength": 0.75}, ("strength", [0.0, 0.2, 0.4, 0.6, 0.75, 0.9])),
    "jpeg":            (jpeg, {"quality": 6}, ("quality", [90, 50, 30, 15, 8, 4, 2])),
}


def degrade(img, name, **overrides):
    func, defaults, _ = DEGRADATIONS[name]
    return func(img, **{**defaults, **overrides})


# ---------------------------------------------------------------------------
# 2. Enhancements
# ---------------------------------------------------------------------------

def gamma_correct(img, gamma):
    """Power-law point transform out = in^gamma. gamma < 1 brightens, > 1 darkens."""
    table = (np.arange(256) / 255.0) ** gamma * 255.0
    return cv2.LUT(img, _lut(table))


def auto_gamma(img, target=0.45):
    """Choose gamma so that the mean luminance maps to `target`: gamma = log(t) / log(mean)."""
    mean = np.clip(brightness(img), 0.02, 0.98)
    gamma = np.log(target) / np.log(mean)
    return gamma_correct(img, float(np.clip(gamma, 0.2, 3.0)))


def hist_equalize(img):
    """Global histogram equalization of the luma channel (colour preserved)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip_limit=2.0, tile=8):
    """Contrast-limited adaptive histogram equalization on the L channel of Lab."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise_gaussian(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(img, (int(ksize), int(ksize)), sigma)


def denoise_median(img, ksize=5):
    return cv2.medianBlur(img, int(ksize))


def denoise_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """Edge-preserving: weights depend on both spatial distance and intensity difference."""
    return cv2.bilateralFilter(img, int(d), sigma_color, sigma_space)


def denoise_nlm(img, h=10, template=5, search=15):
    """Non-local means: averages patches that look alike anywhere in a search window."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, int(template), int(search))


def unsharp(img, amount=1.0, sigma=2.0):
    """Unsharp masking: out = in + amount * (in - GaussianBlur(in))."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


# Fallback thresholds for adaptive_enhance; the notebook replaces them with values calibrated on clean frames.
DEFAULT_THRESHOLDS = {"noise": 6.0, "dark": 0.35, "bright": 0.65, "contrast": 0.16,
                      "glare": 0.12, "sharpness": 60.0}


def adaptive_enhance(img, return_steps=False, thresholds=None, denoiser="nlm"):
    """Measure the frame, then apply only the corrections it needs.

    Order matters: denoise first (brightening amplifies noise), then fix
    exposure, then local contrast, then sharpen. denoiser="bilateral" is the
    real-time variant (~10x faster than non-local means).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    steps = [("input", img)]
    out = img

    sigma = noise_sigma(out)
    if sigma > t["noise"]:
        if denoiser == "nlm":
            h = float(np.clip(1.1 * sigma, 3, 25))
            out = denoise_nlm(out, h=h)
            steps.append((f"NLM denoise (h={h:.0f})", out))
        else:
            sc = float(np.clip(3.0 * sigma, 20, 150))
            out = denoise_bilateral(out, d=7, sigma_color=sc, sigma_space=7)
            steps.append((f"bilateral denoise (sc={sc:.0f})", out))

    mean = brightness(out)
    if mean < t["dark"] or mean > t["bright"]:
        out = auto_gamma(out)
        steps.append((f"auto gamma (mean {mean:.2f})", out))

    if rms_contrast(out) < t["contrast"] or highlight_fraction(out) > t["glare"] or mean < t["dark"]:
        out = clahe(out, clip_limit=2.0, tile=8)
        steps.append(("CLAHE", out))

    if sharpness(out) < t["sharpness"]:
        out = unsharp(out, amount=0.8, sigma=2.0)
        steps.append(("unsharp mask", out))

    return (out, steps) if return_steps else out


# name -> function(img) with tuned default parameters
METHODS = {
    "none":               lambda img: img,
    "gamma (auto)":       auto_gamma,
    "hist. equalize":     hist_equalize,
    "CLAHE":              clahe,
    "gaussian 5x5":       denoise_gaussian,
    "median 5x5":         denoise_median,
    "bilateral":          denoise_bilateral,
    "NLM":                denoise_nlm,
    "gamma + CLAHE":      lambda img: clahe(auto_gamma(img)),
    "NLM + gamma + CLAHE": lambda img: clahe(auto_gamma(denoise_nlm(img))),
    "adaptive (ours)":    adaptive_enhance,
    "adaptive-fast (ours)": lambda img: adaptive_enhance(img, denoiser="bilateral"),
}


# ---------------------------------------------------------------------------
# 3. Quality metrics
# ---------------------------------------------------------------------------

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def brightness(img):
    """Mean luminance in [0, 1]."""
    return float(_gray(img).mean() / 255.0)


def rms_contrast(img):
    """Standard deviation of luminance in [0, 1]."""
    return float(_gray(img).std() / 255.0)


def highlight_fraction(img, level=240):
    """Fraction of nearly saturated pixels (glare / blown-out background)."""
    return float((_gray(img) >= level).mean())


def noise_sigma(img):
    """Fast noise std estimate (Immerkaer, 1996), in grey levels, averaged over channels.

    Convolves with a Laplacian-difference mask that cancels smooth image
    structure, so what remains is mostly noise. Measured per channel because
    converting to grey would average independent channel noise away.
    """
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    resp = np.abs(cv2.filter2D(img.astype(np.float32), -1, mask)[1:-1, 1:-1])  # filters each channel
    return float(np.sqrt(np.pi / 2) * resp.sum() / (6.0 * (w - 2) * (h - 2) * channels))


def sharpness(img):
    """Variance of the Laplacian: low values mean a blurry frame."""
    return float(cv2.Laplacian(_gray(img), cv2.CV_64F).var())


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref, img):
    return float(structural_similarity(_gray(ref), _gray(img), data_range=255))


def quality_report(img):
    return {"brightness": brightness(img), "contrast": rms_contrast(img),
            "noise_sigma": noise_sigma(img), "sharpness": sharpness(img),
            "highlights": highlight_fraction(img)}
