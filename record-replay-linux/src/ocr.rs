use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DEFAULT_TESSERACT: &str = "tesseract";
const DEFAULT_RAPIDOCR_PYTHON: &str = "python3";
const DEFAULT_PADDLEOCR_PYTHON: &str = "python3";
const DEFAULT_LANGUAGE: &str = "eng";
const DEFAULT_RAPIDOCR_LANGUAGE: &str = "ch";
const DEFAULT_PADDLEOCR_LANGUAGE: &str = "en";
const DEFAULT_PSM: &str = "11";
const DEFAULT_TIMEOUT_MS: u64 = 10_000;
const TESSERACT_BACKEND_NAME: &str = "tesseract-cli";
const RAPIDOCR_BACKEND_NAME: &str = "rapidocr-python";
const PADDLEOCR_BACKEND_NAME: &str = "paddleocr-python";
const AUTO_BACKEND_NAME: &str = "auto";
const RAPIDOCR_JSON_PREFIX: &str = "__CODEX_RAPIDOCR_JSON__";
const PADDLEOCR_JSON_PREFIX: &str = "__CODEX_PADDLEOCR_JSON__";
const MAX_OCR_OBSERVATIONS: usize = 200;
const MAX_OCR_NORMALIZED_TEXT_BYTES: usize = 32 * 1024;
const MAX_OCR_CANDIDATE_TEXT_BYTES: usize = 512;

const RAPIDOCR_CHECK_SCRIPT: &str = r#"
import importlib.metadata as metadata
import json
import sys

try:
    import onnxruntime  # noqa: F401
    from rapidocr import RapidOCR

    lang_type = sys.argv[1] if len(sys.argv) > 1 else "ch"
    engine = RapidOCR(params={"Rec.lang_type": lang_type})
    payload = {
        "rapidocr": metadata.version("rapidocr"),
        "onnxruntime": metadata.version("onnxruntime"),
        "lang_type": str(getattr(engine.cfg.Rec, "lang_type", lang_type)),
    }
    print("__CODEX_RAPIDOCR_JSON__" + json.dumps(payload, ensure_ascii=False))
except Exception as exc:
    payload = {"error": f"{type(exc).__name__}: {exc}"}
    print("__CODEX_RAPIDOCR_JSON__" + json.dumps(payload, ensure_ascii=False))
    raise
"#;

const RAPIDOCR_RUN_SCRIPT: &str = r#"
import json
import sys

from rapidocr import RapidOCR

def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)

if len(sys.argv) != 3:
    raise SystemExit("expected rapidocr language and image path arguments")
lang_type = sys.argv[1]
image_path = sys.argv[2]
engine = RapidOCR(params={"Rec.lang_type": lang_type})
result = engine(image_path)
payload = {
    "boxes": as_list(getattr(result, "boxes", None)),
    "txts": as_list(getattr(result, "txts", None)),
    "scores": as_list(getattr(result, "scores", None)),
    "elapse": getattr(result, "elapse", None),
    "lang_type": str(getattr(engine.cfg.Rec, "lang_type", lang_type)),
}
print("__CODEX_RAPIDOCR_JSON__" + json.dumps(payload, ensure_ascii=False))
"#;

const PADDLEOCR_CHECK_SCRIPT: &str = r#"
import importlib.metadata as metadata
import json

try:
    import paddleocr
    from paddleocr import PaddleOCR  # noqa: F401

    paddleocr_version = getattr(paddleocr, "__version__", None)
    if paddleocr_version is None:
        try:
            paddleocr_version = metadata.version("paddleocr")
        except Exception:
            paddleocr_version = "unknown"
    payload = {
        "paddleocr": paddleocr_version,
    }
    try:
        import paddle

        payload["paddle"] = getattr(paddle, "__version__", "unknown")
        payload["cuda_compiled"] = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
        try:
            payload["cuda_device_count"] = int(paddle.device.cuda.device_count())
        except Exception:
            payload["cuda_device_count"] = None
    except Exception as exc:
        payload["paddle_error"] = f"{type(exc).__name__}: {exc}"
    print("__CODEX_PADDLEOCR_JSON__" + json.dumps(payload, ensure_ascii=False))
except Exception as exc:
    payload = {"error": f"{type(exc).__name__}: {exc}"}
    print("__CODEX_PADDLEOCR_JSON__" + json.dumps(payload, ensure_ascii=False))
    raise
"#;

const PADDLEOCR_RUN_SCRIPT: &str = r#"
import json
import sys

from paddleocr import PaddleOCR

PREFIX = "__CODEX_PADDLEOCR_JSON__"

def optional_arg(index):
    if len(sys.argv) <= index:
        return None
    value = str(sys.argv[index]).strip()
    return value or None

def as_plain(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict, str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(item) for item in value]
    return value

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def box_to_poly(box):
    box = as_plain(box)
    if isinstance(box, list) and len(box) == 4 and all(is_number(item) for item in box):
        x1, y1, x2, y2 = box
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return box

def append_observation(payload, box, text, score):
    if text is None:
        return
    text = str(text).strip()
    if not text:
        return
    payload["boxes"].append(box_to_poly(box) if box is not None else [[0, 0], [0, 0], [0, 0], [0, 0]])
    payload["txts"].append(text)
    try:
        payload["scores"].append(float(score))
    except Exception:
        payload["scores"].append(0.0)

def extract_v3_result(result, payload):
    data = result if isinstance(result, dict) else getattr(result, "json", None)
    if callable(data):
        data = data()
    data = as_plain(data)
    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    if not isinstance(data, dict):
        return False
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []
    for index, text in enumerate(texts):
        append_observation(
            payload,
            boxes[index] if index < len(boxes) else None,
            text,
            scores[index] if index < len(scores) else 0.0,
        )
    return True

def extract_legacy_result(results, payload):
    def looks_like_line(item):
        return (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[1], (list, tuple))
            and len(item[1]) >= 2
            and isinstance(item[1][0], str)
        )

    if isinstance(results, list) and any(looks_like_line(item) for item in results):
        pages = [results]
    else:
        pages = results or []
    for page in pages:
        rows = page if isinstance(page, list) else []
        for item in rows:
            if not looks_like_line(item):
                continue
            append_observation(payload, item[0], item[1][0], item[1][1])

if len(sys.argv) != 6:
    raise SystemExit("expected paddleocr language, device, engine, OCR version, and image path arguments")

lang = optional_arg(1)
device = optional_arg(2)
engine = optional_arg(3)
ocr_version = optional_arg(4)
image_path = sys.argv[5]

common_kwargs = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}
if lang:
    common_kwargs["lang"] = lang
if device:
    common_kwargs["device"] = device
if engine:
    common_kwargs["engine"] = engine
if ocr_version:
    common_kwargs["ocr_version"] = ocr_version

payload = {
    "boxes": [],
    "txts": [],
    "scores": [],
    "lang": lang,
    "device": device,
    "engine": engine,
    "ocr_version": ocr_version,
}

try:
    ocr = PaddleOCR(**common_kwargs)
    if hasattr(ocr, "predict"):
        for item in ocr.predict(
            image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ):
            extract_v3_result(item, payload)
    else:
        extract_legacy_result(ocr.ocr(image_path, cls=False), payload)
except TypeError:
    legacy_kwargs = {}
    if lang:
        legacy_kwargs["lang"] = lang
    if device:
        legacy_kwargs["use_gpu"] = device.startswith("gpu")
    ocr = PaddleOCR(use_angle_cls=False, **legacy_kwargs)
    extract_legacy_result(ocr.ocr(image_path, cls=False), payload)

print(PREFIX + json.dumps(payload, ensure_ascii=False))
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OcrMode {
    Auto,
    Enabled,
    Required,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OcrBackendPreference {
    Auto,
    RapidOcr,
    PaddleOcr,
    Tesseract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OcrPolicy {
    pub(crate) mode: OcrMode,
    pub(crate) backend_preference: OcrBackendPreference,
    tesseract_executable: PathBuf,
    rapidocr_python: PathBuf,
    paddleocr_python: PathBuf,
    pub(crate) language: String,
    pub(crate) rapidocr_language: String,
    pub(crate) paddleocr_language: String,
    pub(crate) paddleocr_device: Option<String>,
    pub(crate) paddleocr_engine: Option<String>,
    pub(crate) paddleocr_version: Option<String>,
    pub(crate) page_segmentation_mode: String,
    pub(crate) timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OcrReadiness {
    pub(crate) enabled: bool,
    pub(crate) available: bool,
    pub(crate) backend: String,
    pub(crate) status: String,
    pub(crate) language: String,
    pub(crate) version: Option<String>,
    pub(crate) dependency_hint: Option<String>,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct OcrFrameResult {
    #[serde(skip)]
    pub(crate) runs_ocr: bool,
    #[serde(skip)]
    pub(crate) status: String,
    #[serde(skip)]
    pub(crate) backend: String,
    #[serde(skip)]
    pub(crate) language: String,
    #[serde(skip)]
    pub(crate) backend_version: Option<String>,
    #[serde(skip)]
    pub(crate) normalized_text: String,
    #[serde(skip)]
    pub(crate) dependency_hint: Option<String>,
    #[serde(skip)]
    pub(crate) error: Option<String>,
    #[serde(skip)]
    pub(crate) duration_ms: u128,
    #[serde(skip)]
    pub(crate) truncated: bool,
    #[serde(skip)]
    pub(crate) suppressed_text_observation_count: usize,
    #[serde(skip)]
    pub(crate) observations: Vec<OcrObservation>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct OcrObservation {
    #[serde(rename = "boundingBox")]
    pub(crate) bounding_box: NormalizedBoundingBox,
    #[serde(rename = "pixelBoundingBox")]
    pub(crate) pixel_bounding_box: PixelBoundingBox,
    #[serde(rename = "topCandidates")]
    pub(crate) top_candidates: Vec<OcrCandidate>,
    pub(crate) normalized_text: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct NormalizedBoundingBox {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) coordinate_space: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct PixelBoundingBox {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coordinate_space: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct OcrCandidate {
    pub(crate) text: String,
    pub(crate) confidence: f64,
}

#[derive(Debug, Clone)]
struct TsvWord {
    page: String,
    block: String,
    paragraph: String,
    line: String,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    confidence: f64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RapidOcrVersions {
    rapidocr: Option<String>,
    onnxruntime: Option<String>,
    lang_type: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PaddleOcrVersions {
    paddleocr: Option<String>,
    paddle: Option<String>,
    cuda_compiled: Option<bool>,
    cuda_device_count: Option<i64>,
    paddle_error: Option<String>,
    error: Option<String>,
}

impl OcrPolicy {
    pub(crate) fn from_env() -> Self {
        let mode = env_value("CODEX_SKYSIGHT_OCR")
            .or_else(|| env_value("CODEX_CHRONICLE_OCR"))
            .map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "disabled" | "off" | "false" | "0" => OcrMode::Disabled,
                "enabled" | "on" | "true" | "1" => OcrMode::Enabled,
                "required" | "require" => OcrMode::Required,
                _ => OcrMode::Auto,
            })
            .unwrap_or(OcrMode::Auto);
        let backend_preference = env_value("CODEX_SKYSIGHT_OCR_BACKEND")
            .or_else(|| env_value("CODEX_CHRONICLE_OCR_BACKEND"))
            .map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "rapidocr" | "rapidocr-python" | "rapid-ocr" => OcrBackendPreference::RapidOcr,
                "paddleocr" | "paddleocr-python" | "paddle-ocr" | "paddle" => {
                    OcrBackendPreference::PaddleOcr
                }
                "tesseract" | "tesseract-cli" => OcrBackendPreference::Tesseract,
                _ => OcrBackendPreference::Auto,
            })
            .unwrap_or(OcrBackendPreference::Auto);
        let tesseract_executable = env_value("CODEX_SKYSIGHT_TESSERACT_PATH")
            .or_else(|| env_value("CODEX_CHRONICLE_TESSERACT_PATH"))
            .unwrap_or_else(|| DEFAULT_TESSERACT.to_string());
        let rapidocr_python = env_value("CODEX_SKYSIGHT_RAPIDOCR_PYTHON")
            .or_else(|| env_value("CODEX_CHRONICLE_RAPIDOCR_PYTHON"))
            .unwrap_or_else(|| DEFAULT_RAPIDOCR_PYTHON.to_string());
        let paddleocr_python = env_value("CODEX_SKYSIGHT_PADDLEOCR_PYTHON")
            .or_else(|| env_value("CODEX_CHRONICLE_PADDLEOCR_PYTHON"))
            .unwrap_or_else(|| DEFAULT_PADDLEOCR_PYTHON.to_string());
        let language =
            env_value("CODEX_SKYSIGHT_OCR_LANG").unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        let rapidocr_language = env_value("CODEX_SKYSIGHT_RAPIDOCR_LANG")
            .or_else(|| env_value("CODEX_CHRONICLE_RAPIDOCR_LANG"))
            .unwrap_or_else(|| DEFAULT_RAPIDOCR_LANGUAGE.to_string());
        let paddleocr_language = env_value("CODEX_SKYSIGHT_PADDLEOCR_LANG")
            .or_else(|| env_value("CODEX_CHRONICLE_PADDLEOCR_LANG"))
            .unwrap_or_else(|| DEFAULT_PADDLEOCR_LANGUAGE.to_string());
        let paddleocr_device = env_value("CODEX_SKYSIGHT_PADDLEOCR_DEVICE")
            .or_else(|| env_value("CODEX_CHRONICLE_PADDLEOCR_DEVICE"));
        let paddleocr_engine = env_value("CODEX_SKYSIGHT_PADDLEOCR_ENGINE")
            .or_else(|| env_value("CODEX_CHRONICLE_PADDLEOCR_ENGINE"));
        let paddleocr_version = env_value("CODEX_SKYSIGHT_PADDLEOCR_VERSION")
            .or_else(|| env_value("CODEX_CHRONICLE_PADDLEOCR_VERSION"));
        let page_segmentation_mode =
            env_value("CODEX_SKYSIGHT_OCR_PSM").unwrap_or_else(|| DEFAULT_PSM.to_string());
        let timeout_ms = env_value("CODEX_SKYSIGHT_OCR_TIMEOUT_MS")
            .or_else(|| env_value("CODEX_CHRONICLE_OCR_TIMEOUT_MS"))
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_TIMEOUT_MS);

        Self {
            mode,
            backend_preference,
            tesseract_executable: PathBuf::from(tesseract_executable),
            rapidocr_python: PathBuf::from(rapidocr_python),
            paddleocr_python: PathBuf::from(paddleocr_python),
            language,
            rapidocr_language,
            paddleocr_language,
            paddleocr_device,
            paddleocr_engine,
            paddleocr_version,
            page_segmentation_mode,
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self.backend_preference {
            OcrBackendPreference::Auto => AUTO_BACKEND_NAME,
            OcrBackendPreference::RapidOcr => RAPIDOCR_BACKEND_NAME,
            OcrBackendPreference::PaddleOcr => PADDLEOCR_BACKEND_NAME,
            OcrBackendPreference::Tesseract => TESSERACT_BACKEND_NAME,
        }
    }

    pub(crate) fn mode_name(&self) -> &'static str {
        match self.mode {
            OcrMode::Auto => "auto",
            OcrMode::Enabled => "enabled",
            OcrMode::Required => "required",
            OcrMode::Disabled => "disabled",
        }
    }

    pub(crate) fn readiness(&self) -> OcrReadiness {
        if self.mode == OcrMode::Disabled {
            return OcrReadiness {
                enabled: false,
                available: false,
                backend: self.backend_name().to_string(),
                status: "disabled".to_string(),
                language: self.language.clone(),
                version: None,
                dependency_hint: None,
                error: None,
            };
        }

        match self.backend_preference {
            OcrBackendPreference::RapidOcr => self.rapidocr_readiness(),
            OcrBackendPreference::PaddleOcr => self.paddleocr_readiness(),
            OcrBackendPreference::Tesseract => self.tesseract_readiness(),
            OcrBackendPreference::Auto => {
                let rapidocr = self.rapidocr_readiness();
                if rapidocr.available {
                    return rapidocr;
                }
                let paddleocr = self.paddleocr_readiness();
                if paddleocr.available {
                    return paddleocr;
                }
                let tesseract = self.tesseract_readiness();
                if tesseract.available {
                    return tesseract;
                }
                OcrReadiness {
                    enabled: true,
                    available: false,
                    backend: AUTO_BACKEND_NAME.to_string(),
                    status: "backend_unavailable".to_string(),
                    language: self.language.clone(),
                    version: None,
                    dependency_hint: Some(auto_dependency_hint(&self.language)),
                    error: Some(format!(
                        "RapidOCR unavailable: {}; PaddleOCR unavailable: {}; Tesseract unavailable: {}",
                        rapidocr
                            .error
                            .unwrap_or_else(|| rapidocr.status.to_string()),
                        paddleocr
                            .error
                            .unwrap_or_else(|| paddleocr.status.to_string()),
                        tesseract
                            .error
                            .unwrap_or_else(|| tesseract.status.to_string())
                    )),
                }
            }
        }
    }

    fn paddleocr_readiness(&self) -> OcrReadiness {
        let child = Command::new(&self.paddleocr_python)
            .arg("-c")
            .arg(PADDLEOCR_CHECK_SCRIPT)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let output = match child {
            Ok(child) => match wait_with_output_timeout(child, self.timeout) {
                Ok(output) => output,
                Err(error) => {
                    return OcrReadiness {
                        enabled: true,
                        available: false,
                        backend: PADDLEOCR_BACKEND_NAME.to_string(),
                        status: "backend_unavailable".to_string(),
                        language: self.paddleocr_language.clone(),
                        version: None,
                        dependency_hint: Some(paddleocr_dependency_hint()),
                        error: Some(error.to_string()),
                    };
                }
            },
            Err(error) => {
                return OcrReadiness {
                    enabled: true,
                    available: false,
                    backend: PADDLEOCR_BACKEND_NAME.to_string(),
                    status: "backend_unavailable".to_string(),
                    language: self.paddleocr_language.clone(),
                    version: None,
                    dependency_hint: Some(paddleocr_dependency_hint()),
                    error: Some(error.to_string()),
                };
            }
        };

        if !output.status.success() {
            return OcrReadiness {
                enabled: true,
                available: false,
                backend: PADDLEOCR_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.paddleocr_language.clone(),
                version: None,
                dependency_hint: Some(paddleocr_dependency_hint()),
                error: Some(
                    parse_paddleocr_versions(&output)
                        .ok()
                        .and_then(|versions| versions.error)
                        .unwrap_or_else(|| {
                            format!("PaddleOCR check exited with {}", output.status)
                        }),
                ),
            };
        }

        match parse_paddleocr_versions(&output) {
            Ok(versions) => {
                let paddle_error = versions.paddle_error.clone();
                let mut parts = vec![format!(
                    "paddleocr {}",
                    versions.paddleocr.unwrap_or_else(|| "unknown".to_string())
                )];
                if let Some(paddle) = versions.paddle {
                    parts.push(format!("paddle {paddle}"));
                }
                if let Some(cuda_compiled) = versions.cuda_compiled {
                    parts.push(format!("cuda_compiled {cuda_compiled}"));
                }
                if let Some(cuda_device_count) = versions.cuda_device_count {
                    parts.push(format!("cuda_devices {cuda_device_count}"));
                }
                if let Some(error) = paddle_error {
                    return OcrReadiness {
                        enabled: true,
                        available: false,
                        backend: PADDLEOCR_BACKEND_NAME.to_string(),
                        status: "backend_unavailable".to_string(),
                        language: self.paddleocr_language.clone(),
                        version: Some(parts.join("; ")),
                        dependency_hint: Some(paddleocr_dependency_hint()),
                        error: Some(format!("Paddle runtime unavailable: {error}")),
                    };
                }
                if let Some(gpu_request) = self
                    .paddleocr_device
                    .as_deref()
                    .and_then(parse_paddleocr_gpu_device_request)
                {
                    let requested_index = match gpu_request {
                        Ok(requested_index) => requested_index,
                        Err(error) => {
                            return OcrReadiness {
                                enabled: true,
                                available: false,
                                backend: PADDLEOCR_BACKEND_NAME.to_string(),
                                status: "backend_unavailable".to_string(),
                                language: self.paddleocr_language.clone(),
                                version: Some(parts.join("; ")),
                                dependency_hint: Some(paddleocr_dependency_hint()),
                                error: Some(error),
                            };
                        }
                    };
                    let cuda_compiled = versions.cuda_compiled.unwrap_or(false);
                    let cuda_device_count = versions.cuda_device_count.unwrap_or(0);
                    if !cuda_compiled || cuda_device_count <= 0 {
                        return OcrReadiness {
                            enabled: true,
                            available: false,
                            backend: PADDLEOCR_BACKEND_NAME.to_string(),
                            status: "backend_unavailable".to_string(),
                            language: self.paddleocr_language.clone(),
                            version: Some(parts.join("; ")),
                            dependency_hint: Some(paddleocr_dependency_hint()),
                            error: Some(format!(
                                "PaddleOCR GPU device {} requested but CUDA readiness is unavailable",
                                self.paddleocr_device.as_deref().unwrap_or("gpu")
                            )),
                        };
                    }
                    if let Some(requested_index) = requested_index {
                        if requested_index >= cuda_device_count {
                            return OcrReadiness {
                                enabled: true,
                                available: false,
                                backend: PADDLEOCR_BACKEND_NAME.to_string(),
                                status: "backend_unavailable".to_string(),
                                language: self.paddleocr_language.clone(),
                                version: Some(parts.join("; ")),
                                dependency_hint: Some(paddleocr_dependency_hint()),
                                error: Some(format!(
                                    "PaddleOCR GPU device gpu:{requested_index} requested but only {cuda_device_count} CUDA device(s) are available"
                                )),
                            };
                        }
                    }
                }
                OcrReadiness {
                    enabled: true,
                    available: true,
                    backend: PADDLEOCR_BACKEND_NAME.to_string(),
                    status: "available".to_string(),
                    language: self.paddleocr_language.clone(),
                    version: Some(parts.join("; ")),
                    dependency_hint: None,
                    error: None,
                }
            }
            Err(error) => OcrReadiness {
                enabled: true,
                available: false,
                backend: PADDLEOCR_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.paddleocr_language.clone(),
                version: None,
                dependency_hint: Some(paddleocr_dependency_hint()),
                error: Some(error.to_string()),
            },
        }
    }

    fn rapidocr_readiness(&self) -> OcrReadiness {
        let child = Command::new(&self.rapidocr_python)
            .arg("-c")
            .arg(RAPIDOCR_CHECK_SCRIPT)
            .arg(&self.rapidocr_language)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let output = match child {
            Ok(child) => match wait_with_output_timeout(child, self.timeout) {
                Ok(output) => output,
                Err(error) => {
                    return OcrReadiness {
                        enabled: true,
                        available: false,
                        backend: RAPIDOCR_BACKEND_NAME.to_string(),
                        status: "backend_unavailable".to_string(),
                        language: self.rapidocr_language.clone(),
                        version: None,
                        dependency_hint: Some(rapidocr_dependency_hint()),
                        error: Some(error.to_string()),
                    };
                }
            },
            Err(error) => {
                return OcrReadiness {
                    enabled: true,
                    available: false,
                    backend: RAPIDOCR_BACKEND_NAME.to_string(),
                    status: "backend_unavailable".to_string(),
                    language: self.rapidocr_language.clone(),
                    version: None,
                    dependency_hint: Some(rapidocr_dependency_hint()),
                    error: Some(error.to_string()),
                };
            }
        };

        if !output.status.success() {
            return OcrReadiness {
                enabled: true,
                available: false,
                backend: RAPIDOCR_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.rapidocr_language.clone(),
                version: None,
                dependency_hint: Some(rapidocr_dependency_hint()),
                error: Some(
                    parse_rapidocr_versions(&output)
                        .ok()
                        .and_then(|versions| versions.error)
                        .unwrap_or_else(|| format!("RapidOCR check exited with {}", output.status)),
                ),
            };
        }

        match parse_rapidocr_versions(&output) {
            Ok(versions) => OcrReadiness {
                enabled: true,
                available: true,
                backend: RAPIDOCR_BACKEND_NAME.to_string(),
                status: "available".to_string(),
                language: versions
                    .lang_type
                    .unwrap_or_else(|| self.rapidocr_language.clone()),
                version: Some(format!(
                    "rapidocr {}; onnxruntime {}",
                    versions.rapidocr.unwrap_or_else(|| "unknown".to_string()),
                    versions
                        .onnxruntime
                        .unwrap_or_else(|| "unknown".to_string())
                )),
                dependency_hint: None,
                error: None,
            },
            Err(error) => OcrReadiness {
                enabled: true,
                available: false,
                backend: RAPIDOCR_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.rapidocr_language.clone(),
                version: None,
                dependency_hint: Some(rapidocr_dependency_hint()),
                error: Some(error.to_string()),
            },
        }
    }

    fn tesseract_readiness(&self) -> OcrReadiness {
        match Command::new(&self.tesseract_executable)
            .arg("--version")
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let version = stdout.lines().next().map(str::trim).map(str::to_string);
                OcrReadiness {
                    enabled: true,
                    available: true,
                    backend: TESSERACT_BACKEND_NAME.to_string(),
                    status: "available".to_string(),
                    language: self.language.clone(),
                    version,
                    dependency_hint: None,
                    error: None,
                }
            }
            Ok(output) => OcrReadiness {
                enabled: true,
                available: false,
                backend: TESSERACT_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.language.clone(),
                version: None,
                dependency_hint: Some(tesseract_dependency_hint(&self.language)),
                error: Some(format!("tesseract --version exited with {}", output.status)),
            },
            Err(error) => OcrReadiness {
                enabled: true,
                available: false,
                backend: TESSERACT_BACKEND_NAME.to_string(),
                status: "backend_unavailable".to_string(),
                language: self.language.clone(),
                version: None,
                dependency_hint: Some(tesseract_dependency_hint(&self.language)),
                error: Some(error.to_string()),
            },
        }
    }
}

impl OcrFrameResult {
    fn skipped(readiness: &OcrReadiness) -> Self {
        Self {
            runs_ocr: false,
            status: readiness.status.clone(),
            backend: readiness.backend.clone(),
            language: readiness.language.clone(),
            backend_version: readiness.version.clone(),
            normalized_text: String::new(),
            dependency_hint: readiness.dependency_hint.clone(),
            error: readiness.error.clone(),
            duration_ms: 0,
            truncated: false,
            suppressed_text_observation_count: 0,
            observations: Vec::new(),
        }
    }

    fn failure(readiness: &OcrReadiness, status: &str, duration_ms: u128, error: String) -> Self {
        Self {
            runs_ocr: true,
            status: status.to_string(),
            backend: readiness.backend.clone(),
            language: readiness.language.clone(),
            backend_version: readiness.version.clone(),
            normalized_text: String::new(),
            dependency_hint: None,
            error: Some(error),
            duration_ms,
            truncated: false,
            suppressed_text_observation_count: 0,
            observations: Vec::new(),
        }
    }

    fn required_unavailable(readiness: &OcrReadiness) -> Self {
        Self {
            runs_ocr: false,
            status: "required_backend_unavailable".to_string(),
            backend: readiness.backend.clone(),
            language: readiness.language.clone(),
            backend_version: readiness.version.clone(),
            normalized_text: String::new(),
            dependency_hint: readiness.dependency_hint.clone(),
            error: Some(
                readiness
                    .error
                    .clone()
                    .unwrap_or_else(|| "required OCR backend is unavailable".to_string()),
            ),
            duration_ms: 0,
            truncated: false,
            suppressed_text_observation_count: 0,
            observations: Vec::new(),
        }
    }

    pub(crate) fn apply_text_exclusions(&mut self, exclusion_values: &[String]) {
        if self.normalized_text.trim().is_empty() {
            return;
        }
        if !exclusion_values
            .iter()
            .filter(|value| !value.trim().is_empty())
            .any(|value| contains_case_insensitive(&self.normalized_text, value))
        {
            return;
        }

        self.status = "suppressed_by_exclusion_text".to_string();
        self.suppressed_text_observation_count = self.observations.len();
        self.normalized_text.clear();
        self.observations.clear();
    }

    fn apply_persistence_limits(&mut self) {
        if self.observations.len() > MAX_OCR_OBSERVATIONS {
            self.observations.truncate(MAX_OCR_OBSERVATIONS);
            self.truncated = true;
        }

        let mut truncated_text = false;
        for observation in &mut self.observations {
            truncated_text |= truncate_string_to_bytes(
                &mut observation.normalized_text,
                MAX_OCR_CANDIDATE_TEXT_BYTES,
            );
            for candidate in &mut observation.top_candidates {
                truncated_text |=
                    truncate_string_to_bytes(&mut candidate.text, MAX_OCR_CANDIDATE_TEXT_BYTES);
            }
        }
        if truncated_text {
            self.truncated = true;
        }

        if !self.observations.is_empty() {
            self.normalized_text = self
                .observations
                .iter()
                .map(|observation| observation.normalized_text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
        }
        if truncate_string_to_bytes(&mut self.normalized_text, MAX_OCR_NORMALIZED_TEXT_BYTES) {
            self.truncated = true;
        }
    }

    pub(crate) fn to_json_line(
        &self,
        captured_at: &str,
        frame_index: u64,
        persisted_frame_path: &Path,
        latest_frame_path: &Path,
        display_id: &str,
    ) -> Value {
        let mut value = json!({
            "schema_version": 1,
            "captured_at": captured_at,
            "frame_index": frame_index,
            "persisted_frame_path": persisted_frame_path,
            "latest_frame_path": latest_frame_path,
            "display_id": display_id,
            "normalized_text": self.normalized_text,
            "normalized_text_bytes": self.normalized_text.len(),
            "runs_ocr": self.runs_ocr,
            "ocr_status": self.status,
            "ocr_backend": self.backend,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "observations": self.observations,
        });
        if let Some(version) = &self.backend_version {
            value["ocr_backend_version"] = json!(version);
        }
        if let Some(hint) = &self.dependency_hint {
            value["dependency_hint"] = json!(hint);
        }
        if let Some(error) = &self.error {
            value["error"] = json!(error);
        }
        if self.suppressed_text_observation_count > 0 {
            value["suppressed_text_observation_count"] =
                json!(self.suppressed_text_observation_count);
        }
        value
    }

    #[cfg(test)]
    fn completed_for_test(text: &str) -> Self {
        Self {
            runs_ocr: true,
            status: "completed".to_string(),
            backend: "tesseract-cli".to_string(),
            language: "eng".to_string(),
            backend_version: None,
            normalized_text: text.to_string(),
            dependency_hint: None,
            error: None,
            duration_ms: 1,
            truncated: false,
            suppressed_text_observation_count: 0,
            observations: vec![OcrObservation {
                bounding_box: NormalizedBoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    coordinate_space: "vision-normalized-bottom-left",
                },
                pixel_bounding_box: PixelBoundingBox {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    coordinate_space: "pixel-top-left",
                },
                top_candidates: vec![OcrCandidate {
                    text: text.to_string(),
                    confidence: 1.0,
                }],
                normalized_text: text.to_string(),
            }],
        }
    }
}

#[cfg(test)]
pub(crate) fn recognize_frame(
    policy: &OcrPolicy,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
    exclusion_values: &[String],
) -> OcrFrameResult {
    let readiness = policy.readiness();
    recognize_frame_with_readiness(
        policy,
        &readiness,
        frame_path,
        image_width,
        image_height,
        exclusion_values,
    )
}

pub(crate) fn recognize_frame_with_readiness(
    policy: &OcrPolicy,
    readiness: &OcrReadiness,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
    exclusion_values: &[String],
) -> OcrFrameResult {
    if policy.mode == OcrMode::Disabled {
        return OcrFrameResult::skipped(readiness);
    }
    if !readiness.available {
        return if policy.mode == OcrMode::Required {
            OcrFrameResult::required_unavailable(readiness)
        } else {
            OcrFrameResult::skipped(readiness)
        };
    }

    let started = Instant::now();
    match recognize_frame_inner(policy, readiness, frame_path, image_width, image_height) {
        Ok(mut result) => {
            result.backend_version = readiness.version.clone();
            result.apply_text_exclusions(exclusion_values);
            result.apply_persistence_limits();
            result
        }
        Err(error) => OcrFrameResult::failure(
            readiness,
            if started.elapsed() >= policy.timeout {
                "timeout"
            } else {
                "error"
            },
            started.elapsed().as_millis(),
            error.to_string(),
        ),
    }
}

fn recognize_frame_inner(
    policy: &OcrPolicy,
    readiness: &OcrReadiness,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
) -> Result<OcrFrameResult> {
    match readiness.backend.as_str() {
        RAPIDOCR_BACKEND_NAME => {
            recognize_frame_rapidocr(policy, frame_path, image_width, image_height)
        }
        PADDLEOCR_BACKEND_NAME => {
            recognize_frame_paddleocr(policy, frame_path, image_width, image_height)
        }
        TESSERACT_BACKEND_NAME => {
            recognize_frame_tesseract(policy, frame_path, image_width, image_height)
        }
        backend => anyhow::bail!("OCR backend `{backend}` is not runnable"),
    }
}

fn recognize_frame_tesseract(
    policy: &OcrPolicy,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
) -> Result<OcrFrameResult> {
    let started = Instant::now();
    let temp_dir = frame_path
        .parent()
        .unwrap_or_else(|| Path::new("/tmp"))
        .join(format!(".ocr-{}-{}", std::process::id(), unique_nanos()));
    crate::secure_fs::create_private_dir_all(&temp_dir)?;
    let output_base = temp_dir.join("frame");
    let mut child = Command::new(&policy.tesseract_executable)
        .arg(frame_path)
        .arg(&output_base)
        .arg("-l")
        .arg(&policy.language)
        .arg("--psm")
        .arg(&policy.page_segmentation_mode)
        .arg("tsv")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to start {}", policy.tesseract_executable.display()))?;

    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= policy.timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_dir_all(&temp_dir);
            anyhow::bail!(
                "tesseract timed out after {} ms",
                policy.timeout.as_millis()
            );
        }
        thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        let _ = fs::remove_dir_all(&temp_dir);
        anyhow::bail!("tesseract exited with {status}");
    }

    let tsv_path = output_base.with_extension("tsv");
    let tsv = fs::read_to_string(&tsv_path)
        .with_context(|| format!("failed to read {}", tsv_path.display()))?;
    let observations = observations_from_tsv(&tsv, image_width, image_height)?;
    let normalized_text = observations
        .iter()
        .map(|observation| observation.normalized_text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let _ = fs::remove_dir_all(&temp_dir);

    Ok(OcrFrameResult {
        runs_ocr: true,
        status: "completed".to_string(),
        backend: TESSERACT_BACKEND_NAME.to_string(),
        language: policy.language.clone(),
        backend_version: None,
        normalized_text,
        dependency_hint: None,
        error: None,
        duration_ms: started.elapsed().as_millis(),
        truncated: false,
        suppressed_text_observation_count: 0,
        observations,
    })
}

fn recognize_frame_rapidocr(
    policy: &OcrPolicy,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
) -> Result<OcrFrameResult> {
    let started = Instant::now();
    let child = Command::new(&policy.rapidocr_python)
        .arg("-c")
        .arg(RAPIDOCR_RUN_SCRIPT)
        .arg(&policy.rapidocr_language)
        .arg(frame_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start {}", policy.rapidocr_python.display()))?;
    let output = wait_with_output_timeout(child, policy.timeout)?;
    if !output.status.success() {
        anyhow::bail!("RapidOCR exited with {}", output.status);
    }

    let payload = parse_rapidocr_json_line(&output)?;
    let observations = observations_from_rapidocr_json(&payload, image_width, image_height)?;
    let normalized_text = observations
        .iter()
        .map(|observation| observation.normalized_text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(OcrFrameResult {
        runs_ocr: true,
        status: "completed".to_string(),
        backend: RAPIDOCR_BACKEND_NAME.to_string(),
        language: policy.rapidocr_language.clone(),
        backend_version: None,
        normalized_text,
        dependency_hint: None,
        error: None,
        duration_ms: started.elapsed().as_millis(),
        truncated: false,
        suppressed_text_observation_count: 0,
        observations,
    })
}

fn recognize_frame_paddleocr(
    policy: &OcrPolicy,
    frame_path: &Path,
    image_width: u32,
    image_height: u32,
) -> Result<OcrFrameResult> {
    let started = Instant::now();
    let child = Command::new(&policy.paddleocr_python)
        .arg("-c")
        .arg(PADDLEOCR_RUN_SCRIPT)
        .arg(&policy.paddleocr_language)
        .arg(policy.paddleocr_device.as_deref().unwrap_or(""))
        .arg(policy.paddleocr_engine.as_deref().unwrap_or(""))
        .arg(policy.paddleocr_version.as_deref().unwrap_or(""))
        .arg(frame_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start {}", policy.paddleocr_python.display()))?;
    let output = wait_with_output_timeout(child, policy.timeout)?;
    if !output.status.success() {
        anyhow::bail!("PaddleOCR exited with {}", output.status);
    }

    let payload = parse_paddleocr_json_line(&output)?;
    let observations = observations_from_rapidocr_json(&payload, image_width, image_height)?;
    let normalized_text = observations
        .iter()
        .map(|observation| observation.normalized_text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(OcrFrameResult {
        runs_ocr: true,
        status: "completed".to_string(),
        backend: PADDLEOCR_BACKEND_NAME.to_string(),
        language: policy.paddleocr_language.clone(),
        backend_version: None,
        normalized_text,
        dependency_hint: None,
        error: None,
        duration_ms: started.elapsed().as_millis(),
        truncated: false,
        suppressed_text_observation_count: 0,
        observations,
    })
}

fn observations_from_tsv(
    tsv: &str,
    image_width: u32,
    image_height: u32,
) -> Result<Vec<OcrObservation>> {
    let mut lines = tsv.lines();
    let Some(header) = lines.next() else {
        return Ok(Vec::new());
    };
    let columns = header.split('\t').collect::<Vec<_>>();
    let index = |name: &str| {
        columns
            .iter()
            .position(|column| *column == name)
            .with_context(|| format!("missing TSV column {name}"))
    };
    let level_idx = index("level")?;
    let page_idx = index("page_num")?;
    let block_idx = index("block_num")?;
    let paragraph_idx = index("par_num")?;
    let line_idx = index("line_num")?;
    let left_idx = index("left")?;
    let top_idx = index("top")?;
    let width_idx = index("width")?;
    let height_idx = index("height")?;
    let confidence_idx = index("conf")?;
    let text_idx = index("text")?;

    let mut words = Vec::new();
    for line in lines {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() <= text_idx || fields.get(level_idx) != Some(&"5") {
            continue;
        }
        let text = fields[text_idx].trim();
        if text.is_empty() {
            continue;
        }
        words.push(TsvWord {
            page: fields[page_idx].to_string(),
            block: fields[block_idx].to_string(),
            paragraph: fields[paragraph_idx].to_string(),
            line: fields[line_idx].to_string(),
            left: parse_u32(fields[left_idx]),
            top: parse_u32(fields[top_idx]),
            width: parse_u32(fields[width_idx]),
            height: parse_u32(fields[height_idx]),
            confidence: parse_confidence(fields[confidence_idx]),
            text: text.to_string(),
        });
    }

    let mut observations = Vec::new();
    let mut current_key: Option<(String, String, String, String)> = None;
    let mut current_words: Vec<TsvWord> = Vec::new();
    for word in words {
        let key = (
            word.page.clone(),
            word.block.clone(),
            word.paragraph.clone(),
            word.line.clone(),
        );
        if current_key
            .as_ref()
            .is_some_and(|existing| *existing != key)
        {
            observations.push(observation_from_words(
                &current_words,
                image_width,
                image_height,
            ));
            current_words.clear();
        }
        current_key = Some(key);
        current_words.push(word);
    }
    if !current_words.is_empty() {
        observations.push(observation_from_words(
            &current_words,
            image_width,
            image_height,
        ));
    }

    Ok(observations)
}

fn observation_from_words(
    words: &[TsvWord],
    image_width: u32,
    image_height: u32,
) -> OcrObservation {
    let x1 = words.iter().map(|word| word.left).min().unwrap_or(0);
    let y1 = words.iter().map(|word| word.top).min().unwrap_or(0);
    let x2 = words
        .iter()
        .map(|word| word.left.saturating_add(word.width))
        .max()
        .unwrap_or(x1);
    let y2 = words
        .iter()
        .map(|word| word.top.saturating_add(word.height))
        .max()
        .unwrap_or(y1);
    let width = x2.saturating_sub(x1);
    let height = y2.saturating_sub(y1);
    let text = words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let confidence = if words.is_empty() {
        0.0
    } else {
        words.iter().map(|word| word.confidence).sum::<f64>() / words.len() as f64
    };

    OcrObservation {
        bounding_box: NormalizedBoundingBox {
            x: if image_width == 0 {
                0.0
            } else {
                x1 as f64 / image_width as f64
            },
            y: if image_height == 0 {
                0.0
            } else {
                1.0 - (y2 as f64 / image_height as f64)
            },
            width: if image_width == 0 {
                0.0
            } else {
                width as f64 / image_width as f64
            },
            height: if image_height == 0 {
                0.0
            } else {
                height as f64 / image_height as f64
            },
            coordinate_space: "vision-normalized-bottom-left",
        },
        pixel_bounding_box: PixelBoundingBox {
            x: x1,
            y: y1,
            width,
            height,
            coordinate_space: "pixel-top-left",
        },
        top_candidates: vec![OcrCandidate {
            text: text.clone(),
            confidence,
        }],
        normalized_text: text,
    }
}

fn observations_from_rapidocr_json(
    payload: &Value,
    image_width: u32,
    image_height: u32,
) -> Result<Vec<OcrObservation>> {
    let boxes = payload
        .get("boxes")
        .and_then(Value::as_array)
        .context("RapidOCR output missing boxes")?;
    let txts = payload
        .get("txts")
        .and_then(Value::as_array)
        .context("RapidOCR output missing txts")?;
    let scores = payload
        .get("scores")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut observations = Vec::new();
    for (index, text_value) in txts.iter().enumerate() {
        let text = text_value
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let Some(box_value) = boxes.get(index) else {
            continue;
        };
        let Some((x1, y1, x2, y2)) = rapidocr_box_bounds(box_value) else {
            continue;
        };
        let confidence = scores
            .get(index)
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        observations.push(observation_from_rect(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
            text,
            confidence,
        ));
    }

    Ok(observations)
}

fn rapidocr_box_bounds(value: &Value) -> Option<(f64, f64, f64, f64)> {
    let points = value.as_array()?;
    if points.len() == 4 && points.iter().all(|point| point.as_f64().is_some()) {
        let x1 = points.first()?.as_f64()?;
        let y1 = points.get(1)?.as_f64()?;
        let x2 = points.get(2)?.as_f64()?;
        let y2 = points.get(3)?.as_f64()?;
        return Some((x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)));
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for point in points {
        let coordinates = point.as_array()?;
        let x = coordinates.first()?.as_f64()?;
        let y = coordinates.get(1)?.as_f64()?;
        if x.is_finite() && y.is_finite() {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.is_empty() || ys.is_empty() {
        return None;
    }
    Some((
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        ys.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ))
}

#[allow(clippy::too_many_arguments)]
fn observation_from_rect(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    image_width: u32,
    image_height: u32,
    text: &str,
    confidence: f64,
) -> OcrObservation {
    let max_x = if image_width == 0 {
        f64::MAX
    } else {
        image_width as f64
    };
    let max_y = if image_height == 0 {
        f64::MAX
    } else {
        image_height as f64
    };
    let x1 = x1.clamp(0.0, max_x);
    let y1 = y1.clamp(0.0, max_y);
    let x2 = x2.clamp(x1, max_x);
    let y2 = y2.clamp(y1, max_y);
    let pixel_x = x1.floor() as u32;
    let pixel_y = y1.floor() as u32;
    let pixel_width = (x2 - x1).max(0.0).min(u32::MAX as f64) as u32;
    let pixel_height = (y2 - y1).max(0.0).min(u32::MAX as f64) as u32;
    let text = text.to_string();

    OcrObservation {
        bounding_box: NormalizedBoundingBox {
            x: if image_width == 0 {
                0.0
            } else {
                x1 / image_width as f64
            },
            y: if image_height == 0 {
                0.0
            } else {
                1.0 - (y2 / image_height as f64)
            },
            width: if image_width == 0 {
                0.0
            } else {
                (x2 - x1) / image_width as f64
            },
            height: if image_height == 0 {
                0.0
            } else {
                (y2 - y1) / image_height as f64
            },
            coordinate_space: "vision-normalized-bottom-left",
        },
        pixel_bounding_box: PixelBoundingBox {
            x: pixel_x,
            y: pixel_y,
            width: pixel_width,
            height: pixel_height,
            coordinate_space: "pixel-top-left",
        },
        top_candidates: vec![OcrCandidate {
            text: text.clone(),
            confidence,
        }],
        normalized_text: text,
    }
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn rapidocr_dependency_hint() -> String {
    "Install Python packages `rapidocr` and `onnxruntime` plus the OpenCV runtime dependency `libGL.so.1`, or set CODEX_SKYSIGHT_OCR_BACKEND=tesseract".to_string()
}

fn paddleocr_dependency_hint() -> String {
    "Install Python package `paddleocr` plus a matching PaddlePaddle runtime; for CUDA, install the GPU PaddlePaddle wheel and set CODEX_SKYSIGHT_PADDLEOCR_DEVICE=gpu:0".to_string()
}

fn parse_paddleocr_gpu_device_request(device: &str) -> Option<Result<Option<i64>, String>> {
    let trimmed = device.trim();
    let normalized = trimmed.to_ascii_lowercase();
    if normalized == "gpu" {
        return Some(Ok(None));
    }
    let raw_index = normalized.strip_prefix("gpu:")?;
    if raw_index.is_empty() {
        return Some(Err(format!("Malformed PaddleOCR GPU device `{trimmed}`")));
    }
    match raw_index.parse::<i64>() {
        Ok(index) if index >= 0 => Some(Ok(Some(index))),
        _ => Some(Err(format!("Malformed PaddleOCR GPU device `{trimmed}`"))),
    }
}

fn tesseract_dependency_hint(language: &str) -> String {
    format!("Install tesseract and traineddata for language `{language}`")
}

fn auto_dependency_hint(language: &str) -> String {
    format!(
        "{}; optional higher-quality backend: {}; fallback: {}",
        rapidocr_dependency_hint(),
        paddleocr_dependency_hint(),
        tesseract_dependency_hint(language)
    )
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn truncate_string_to_bytes(value: &mut String, max_bytes: usize) -> bool {
    if value.len() <= max_bytes {
        return false;
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    true
}

fn wait_with_output_timeout(mut child: Child, timeout: Duration) -> Result<Output> {
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child
                .wait_with_output()
                .context("failed to collect OCR output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("OCR process timed out after {} ms", timeout.as_millis());
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn parse_rapidocr_versions(output: &Output) -> Result<RapidOcrVersions> {
    let payload = parse_rapidocr_json_line(output)?;
    serde_json::from_value(payload).context("failed to parse RapidOCR version payload")
}

fn parse_rapidocr_json_line(output: &Output) -> Result<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(RAPIDOCR_JSON_PREFIX))
        .context("RapidOCR JSON payload was not found on stdout")?;
    serde_json::from_str(line).context("failed to parse RapidOCR JSON payload")
}

fn parse_paddleocr_versions(output: &Output) -> Result<PaddleOcrVersions> {
    let payload = parse_paddleocr_json_line(output)?;
    serde_json::from_value(payload).context("failed to parse PaddleOCR version payload")
}

fn parse_paddleocr_json_line(output: &Output) -> Result<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(PADDLEOCR_JSON_PREFIX))
        .context("PaddleOCR JSON payload was not found on stdout")?;
    serde_json::from_str(line).context("failed to parse PaddleOCR JSON payload")
}

fn parse_u32(value: &str) -> u32 {
    value
        .parse::<i64>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(0) as u32
}

fn parse_confidence(value: &str) -> f64 {
    value
        .parse::<f64>()
        .ok()
        .filter(|value| *value >= 0.0)
        .map(|value| (value / 100.0).min(1.0))
        .unwrap_or(0.0)
}

fn unique_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};

    #[test]
    fn policy_defaults_to_auto_backend_preference() {
        let _guard = env_lock();
        clear_ocr_env();

        let policy = OcrPolicy::from_env();

        assert_eq!(policy.mode, OcrMode::Auto);
        assert_eq!(policy.backend_preference, OcrBackendPreference::Auto);
        assert_eq!(policy.backend_name(), "auto");
        assert_eq!(policy.language, "eng");
        assert_eq!(policy.rapidocr_language, "ch");
        assert_eq!(policy.paddleocr_language, "en");
        assert_eq!(policy.paddleocr_device, None);
        assert_eq!(policy.paddleocr_engine, None);
        assert_eq!(policy.paddleocr_version, None);
        assert_eq!(policy.page_segmentation_mode, "11");
        assert_eq!(policy.timeout.as_secs(), 10);
    }

    #[test]
    fn rapidocr_run_script_requires_language_and_image_path() {
        assert!(RAPIDOCR_RUN_SCRIPT.contains("len(sys.argv) != 3"));
        assert!(RAPIDOCR_RUN_SCRIPT.contains("expected rapidocr language and image path"));
        assert!(!RAPIDOCR_RUN_SCRIPT.contains("else sys.argv[1]"));
    }

    #[test]
    fn paddleocr_backend_preference_reads_runtime_options() {
        let _guard = env_lock();
        clear_ocr_env();
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "paddleocr");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_LANG", "pt");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_DEVICE", "gpu:0");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_ENGINE", "paddle_static");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_VERSION", "PP-OCRv5");

        let policy = OcrPolicy::from_env();

        assert_eq!(policy.backend_preference, OcrBackendPreference::PaddleOcr);
        assert_eq!(policy.backend_name(), "paddleocr-python");
        assert_eq!(policy.paddleocr_language, "pt");
        assert_eq!(policy.paddleocr_device.as_deref(), Some("gpu:0"));
        assert_eq!(policy.paddleocr_engine.as_deref(), Some("paddle_static"));
        assert_eq!(policy.paddleocr_version.as_deref(), Some("PP-OCRv5"));
    }

    #[test]
    fn missing_backend_reports_unavailable_without_requiring_host_dependency() {
        let _guard = env_lock();
        clear_ocr_env();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "tesseract");
        std::env::set_var(
            "CODEX_SKYSIGHT_TESSERACT_PATH",
            "/definitely/missing/tesseract",
        );

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert!(readiness.enabled);
        assert!(!readiness.available);
        assert_eq!(readiness.status, "backend_unavailable");
        assert!(readiness.dependency_hint.is_some());
    }

    #[test]
    fn required_mode_marks_missing_backend_as_required_failure() {
        let _guard = env_lock();
        clear_ocr_env();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "required");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "tesseract");
        std::env::set_var(
            "CODEX_SKYSIGHT_TESSERACT_PATH",
            "/definitely/missing/tesseract",
        );

        let policy = OcrPolicy::from_env();
        let result = recognize_frame(&policy, Path::new("/tmp/missing-frame.jpg"), 100, 50, &[]);

        assert!(!result.runs_ocr);
        assert_eq!(result.status, "required_backend_unavailable");
        assert!(result.error.is_some());
        assert!(result.dependency_hint.is_some());
    }

    #[test]
    fn fake_rapidocr_backend_parses_json_observations() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 3 ]]; then
  echo '__CODEX_RAPIDOCR_JSON__{"rapidocr":"3.9.1","onnxruntime":"1.22.0","lang_type":"en"}'
  exit 0
fi
echo '__CODEX_RAPIDOCR_JSON__{"boxes":[[[10,20],[90,20],[90,40],[10,40]],[[10,60],[55,60],[55,74],[10,74]]],"txts":["Codex Rapid","OCR"],"scores":[0.97,0.91],"elapse":0.123,"lang_type":"en"}'
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_python, permissions).unwrap();
        let image = temp.path().join("frame.jpg");
        fs::write(&image, b"not-a-real-image-but-fake-backend-does-not-care").unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "rapidocr");
        std::env::set_var("CODEX_SKYSIGHT_RAPIDOCR_PYTHON", &fake_python);
        std::env::set_var("CODEX_SKYSIGHT_RAPIDOCR_LANG", "en");

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();
        assert!(readiness.available);
        assert_eq!(readiness.backend, "rapidocr-python");
        assert_eq!(readiness.language, "en");
        assert!(readiness
            .version
            .as_deref()
            .is_some_and(|version| version.contains("rapidocr 3.9.1")));

        let result = recognize_frame(&policy, &image, 200, 100, &[]);

        assert_eq!(result.status, "completed");
        assert!(result.runs_ocr);
        assert_eq!(result.backend, "rapidocr-python");
        assert_eq!(result.language, "en");
        assert_eq!(result.normalized_text, "Codex Rapid\nOCR");
        assert_eq!(result.observations.len(), 2);
        assert_eq!(result.observations[0].pixel_bounding_box.x, 10);
        assert_eq!(result.observations[0].pixel_bounding_box.y, 20);
        assert_eq!(result.observations[0].pixel_bounding_box.width, 80);
        assert_eq!(result.observations[0].pixel_bounding_box.height, 20);
        assert!(result.observations[0].top_candidates[0].confidence > 0.9);
    }

    #[test]
    fn fake_paddleocr_backend_parses_json_observations() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 2 ]]; then
  echo '__CODEX_PADDLEOCR_JSON__{"paddleocr":"3.7.0","paddle":"3.0.0","cuda_compiled":true,"cuda_device_count":1}'
  exit 0
fi
echo '__CODEX_PADDLEOCR_JSON__{"boxes":[[[12,18],[92,18],[92,38],[12,38]],[20,60,80,76]],"txts":["Codex Paddle","GPU"],"scores":[0.98,0.94],"lang":"pt","device":"gpu:0"}'
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_python, permissions).unwrap();
        let image = temp.path().join("frame.jpg");
        fs::write(&image, b"not-a-real-image-but-fake-backend-does-not-care").unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "paddleocr");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_PYTHON", &fake_python);
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_LANG", "pt");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_DEVICE", "gpu:0");

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();
        assert!(readiness.available);
        assert_eq!(readiness.backend, "paddleocr-python");
        assert_eq!(readiness.language, "pt");
        assert!(readiness
            .version
            .as_deref()
            .is_some_and(|version| version.contains("paddleocr 3.7.0")));

        let result = recognize_frame(&policy, &image, 200, 100, &[]);

        assert_eq!(result.status, "completed");
        assert!(result.runs_ocr);
        assert_eq!(result.backend, "paddleocr-python");
        assert_eq!(result.language, "pt");
        assert_eq!(result.normalized_text, "Codex Paddle\nGPU");
        assert_eq!(result.observations.len(), 2);
        assert_eq!(result.observations[0].pixel_bounding_box.x, 12);
        assert_eq!(result.observations[0].pixel_bounding_box.y, 18);
        assert_eq!(result.observations[0].pixel_bounding_box.width, 80);
        assert_eq!(result.observations[0].pixel_bounding_box.height, 20);
        assert_eq!(result.observations[1].pixel_bounding_box.x, 20);
        assert_eq!(result.observations[1].pixel_bounding_box.width, 60);
        assert!(result.observations[0].top_candidates[0].confidence > 0.9);
    }

    #[test]
    fn paddleocr_readiness_rejects_paddle_runtime_errors() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo '__CODEX_PADDLEOCR_JSON__{"paddleocr":"3.7.0","paddle_error":"libcuda.so not found"}'
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_python, permissions).unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "paddleocr");
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_PYTHON", &fake_python);

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert!(!readiness.available);
        assert_eq!(readiness.backend, "paddleocr-python");
        assert_eq!(readiness.status, "backend_unavailable");
        assert!(readiness.dependency_hint.is_some());
        assert!(readiness
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Paddle runtime unavailable")));
    }

    #[test]
    fn auto_backend_falls_back_when_requested_paddleocr_gpu_is_unavailable() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo '__CODEX_PADDLEOCR_JSON__{"paddleocr":"3.7.0","paddle":"3.0.0","cuda_compiled":false,"cuda_device_count":0}'
"#,
        )
        .unwrap();
        let fake_tesseract = temp.path().join("fake-tesseract");
        fs::write(
            &fake_tesseract,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "tesseract 5.3.4"
  exit 0
fi
exit 1
"#,
        )
        .unwrap();
        for path in [&fake_python, &fake_tesseract] {
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).unwrap();
        }
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var(
            "CODEX_SKYSIGHT_RAPIDOCR_PYTHON",
            "/definitely/missing/python-for-rapidocr",
        );
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_PYTHON", &fake_python);
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_DEVICE", "gpu:0");
        std::env::set_var("CODEX_SKYSIGHT_TESSERACT_PATH", &fake_tesseract);

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert_eq!(policy.backend_preference, OcrBackendPreference::Auto);
        assert!(readiness.available);
        assert_eq!(readiness.backend, "tesseract-cli");
    }

    #[test]
    fn auto_backend_falls_back_when_requested_paddleocr_gpu_index_is_unavailable() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo '__CODEX_PADDLEOCR_JSON__{"paddleocr":"3.7.0","paddle":"3.0.0","cuda_compiled":true,"cuda_device_count":1}'
"#,
        )
        .unwrap();
        let fake_tesseract = temp.path().join("fake-tesseract");
        fs::write(
            &fake_tesseract,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "tesseract 5.3.4"
  exit 0
fi
exit 1
"#,
        )
        .unwrap();
        for path in [&fake_python, &fake_tesseract] {
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).unwrap();
        }
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var(
            "CODEX_SKYSIGHT_RAPIDOCR_PYTHON",
            "/definitely/missing/python-for-rapidocr",
        );
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_PYTHON", &fake_python);
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_DEVICE", "gpu:1");
        std::env::set_var("CODEX_SKYSIGHT_TESSERACT_PATH", &fake_tesseract);

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert_eq!(policy.backend_preference, OcrBackendPreference::Auto);
        assert!(readiness.available);
        assert_eq!(readiness.backend, "tesseract-cli");
    }

    #[test]
    fn auto_backend_prefers_rapidocr_when_available() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo '__CODEX_RAPIDOCR_JSON__{"rapidocr":"3.9.1","onnxruntime":"1.22.0"}'
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_python, permissions).unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_RAPIDOCR_PYTHON", &fake_python);

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert_eq!(policy.backend_preference, OcrBackendPreference::Auto);
        assert!(readiness.available);
        assert_eq!(readiness.backend, "rapidocr-python");
    }

    #[test]
    fn auto_backend_uses_paddleocr_before_tesseract_when_rapidocr_is_unavailable() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python");
        fs::write(
            &fake_python,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo '__CODEX_PADDLEOCR_JSON__{"paddleocr":"3.7.0","paddle":"3.0.0"}'
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_python, permissions).unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var(
            "CODEX_SKYSIGHT_RAPIDOCR_PYTHON",
            "/definitely/missing/python-for-rapidocr",
        );
        std::env::set_var("CODEX_SKYSIGHT_PADDLEOCR_PYTHON", &fake_python);

        let policy = OcrPolicy::from_env();
        let readiness = policy.readiness();

        assert_eq!(policy.backend_preference, OcrBackendPreference::Auto);
        assert!(readiness.available);
        assert_eq!(readiness.backend, "paddleocr-python");
    }

    #[test]
    fn fake_tesseract_tsv_groups_words_into_observations() {
        let _guard = env_lock();
        clear_ocr_env();
        let temp = tempfile::tempdir().unwrap();
        let fake_tesseract = temp.path().join("fake-tesseract");
        fs::write(
            &fake_tesseract,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "tesseract 5.3.4"
  exit 0
fi
out="${2}.tsv"
cat > "$out" <<'TSV'
level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
5	1	1	1	1	1	10	20	40	12	95	Codex
5	1	1	1	1	2	55	20	30	12	91	OCR
5	1	1	1	2	1	10	50	35	12	80	Local
TSV
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_tesseract).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_tesseract, permissions).unwrap();
        let image = temp.path().join("frame.jpg");
        fs::write(&image, b"not-a-real-image-but-fake-backend-does-not-care").unwrap();
        std::env::set_var("CODEX_SKYSIGHT_OCR", "enabled");
        std::env::set_var("CODEX_SKYSIGHT_OCR_BACKEND", "tesseract");
        std::env::set_var("CODEX_SKYSIGHT_TESSERACT_PATH", &fake_tesseract);

        let policy = OcrPolicy::from_env();
        let result = recognize_frame(&policy, &image, 200, 100, &[]);

        assert_eq!(result.status, "completed");
        assert!(result.runs_ocr);
        assert_eq!(result.normalized_text, "Codex OCR\nLocal");
        assert_eq!(result.observations.len(), 2);
        assert_eq!(result.observations[0].normalized_text, "Codex OCR");
        assert_eq!(result.observations[0].pixel_bounding_box.x, 10);
        assert_eq!(result.observations[0].pixel_bounding_box.width, 75);
        assert!(result.observations[0].top_candidates[0].confidence > 0.9);
    }

    #[test]
    fn text_matching_exclusion_suppresses_output() {
        let _guard = env_lock();
        clear_ocr_env();
        let mut result = OcrFrameResult::completed_for_test("bank.example account");

        result.apply_text_exclusions(&["bank.example".to_string()]);

        assert_eq!(result.status, "suppressed_by_exclusion_text");
        assert_eq!(result.normalized_text, "");
        assert!(result.observations.is_empty());
        assert_eq!(result.suppressed_text_observation_count, 1);
    }

    #[test]
    fn rapidocr_zero_area_boxes_stay_zero_area_pixels() {
        let payload = serde_json::json!({
            "boxes": [[
                [10.0, 20.0],
                [10.0, 20.0],
                [10.0, 20.0],
                [10.0, 20.0]
            ]],
            "txts": ["dot"],
            "scores": [0.8]
        });

        let observations = observations_from_rapidocr_json(&payload, 100, 50).unwrap();

        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].pixel_bounding_box.x, 10);
        assert_eq!(observations[0].pixel_bounding_box.y, 20);
        assert_eq!(observations[0].pixel_bounding_box.width, 0);
        assert_eq!(observations[0].pixel_bounding_box.height, 0);
    }

    #[test]
    fn persistence_limits_bound_ocr_text_and_observations() {
        let _guard = env_lock();
        clear_ocr_env();
        let mut result = OcrFrameResult {
            runs_ocr: true,
            status: "completed".to_string(),
            backend: "tesseract-cli".to_string(),
            language: "eng".to_string(),
            backend_version: None,
            normalized_text: String::new(),
            dependency_hint: None,
            error: None,
            duration_ms: 1,
            truncated: false,
            suppressed_text_observation_count: 0,
            observations: (0..(MAX_OCR_OBSERVATIONS + 5))
                .map(|index| OcrObservation {
                    bounding_box: NormalizedBoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                        coordinate_space: "vision-normalized-bottom-left",
                    },
                    pixel_bounding_box: PixelBoundingBox {
                        x: 0,
                        y: 0,
                        width: 1,
                        height: 1,
                        coordinate_space: "pixel-top-left",
                    },
                    top_candidates: vec![OcrCandidate {
                        text: format!("{index}-{}", "x".repeat(MAX_OCR_CANDIDATE_TEXT_BYTES + 20)),
                        confidence: 0.9,
                    }],
                    normalized_text: format!(
                        "{index}-{}",
                        "y".repeat(MAX_OCR_CANDIDATE_TEXT_BYTES + 20)
                    ),
                })
                .collect(),
        };

        result.apply_persistence_limits();

        assert!(result.truncated);
        assert_eq!(result.observations.len(), MAX_OCR_OBSERVATIONS);
        assert!(result.normalized_text.len() <= MAX_OCR_NORMALIZED_TEXT_BYTES);
        assert!(result
            .observations
            .iter()
            .all(|observation| observation.normalized_text.len() <= MAX_OCR_CANDIDATE_TEXT_BYTES));
        assert!(result.observations.iter().all(|observation| observation
            .top_candidates
            .iter()
            .all(|candidate| candidate.text.len() <= MAX_OCR_CANDIDATE_TEXT_BYTES)));
    }

    fn clear_ocr_env() {
        for key in [
            "CODEX_SKYSIGHT_OCR",
            "CODEX_CHRONICLE_OCR",
            "CODEX_SKYSIGHT_OCR_BACKEND",
            "CODEX_CHRONICLE_OCR_BACKEND",
            "CODEX_SKYSIGHT_TESSERACT_PATH",
            "CODEX_CHRONICLE_TESSERACT_PATH",
            "CODEX_SKYSIGHT_RAPIDOCR_PYTHON",
            "CODEX_CHRONICLE_RAPIDOCR_PYTHON",
            "CODEX_SKYSIGHT_RAPIDOCR_LANG",
            "CODEX_CHRONICLE_RAPIDOCR_LANG",
            "CODEX_SKYSIGHT_PADDLEOCR_PYTHON",
            "CODEX_CHRONICLE_PADDLEOCR_PYTHON",
            "CODEX_SKYSIGHT_PADDLEOCR_LANG",
            "CODEX_CHRONICLE_PADDLEOCR_LANG",
            "CODEX_SKYSIGHT_PADDLEOCR_DEVICE",
            "CODEX_CHRONICLE_PADDLEOCR_DEVICE",
            "CODEX_SKYSIGHT_PADDLEOCR_ENGINE",
            "CODEX_CHRONICLE_PADDLEOCR_ENGINE",
            "CODEX_SKYSIGHT_PADDLEOCR_VERSION",
            "CODEX_CHRONICLE_PADDLEOCR_VERSION",
            "CODEX_SKYSIGHT_OCR_LANG",
            "CODEX_SKYSIGHT_OCR_PSM",
            "CODEX_SKYSIGHT_OCR_TIMEOUT_MS",
        ] {
            std::env::remove_var(key);
        }
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap()
    }
}
