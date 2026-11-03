"""Model discovery through ComfyUI's model paths, and image preparation.

Everything here is local to this package: it only uses ComfyUI's
``folder_paths`` registry plus Pillow/NumPy, which are already part of the
ComfyUI environment. No GPU work happens in this module, so ComfyUI's own
CUDA device selection is never touched.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import struct
from dataclasses import dataclass

import folder_paths
import numpy as np
from PIL import Image

# Model folder key. extra_model_paths.yaml maps a shared directory to "LLM";
# if the key does not exist yet we register <models_dir>/LLM ourselves.
LLM_FOLDER = "LLM"

# ComfyUI folder keys that are scanned for GGUF files, in priority order.
# "text_encoders" also covers <models_dir>/clip (ComfyUI registers both paths
# under that key), which is where GGUF caption/VL models usually live. The set
# can be extended without touching the code:
#   COMFYUI_LLM_SESSION_MODEL_FOLDERS="LLM,text_encoders,/mnt/other/ggufs"
DEFAULT_MODEL_FOLDERS = (LLM_FOLDER, "text_encoders")
MODEL_FOLDERS_ENV = "COMFYUI_LLM_SESSION_MODEL_FOLDERS"
MODEL_SCAN_MAX_DEPTH = 4

# Placeholder choice that lets a text-only model be used without a projector.
NO_PROJECTOR = "(none)"

# Placeholder accepted by run() for workflows that stored it, and as the fallback
# when nothing matches. It is deliberately NOT offered in the dropdown: the row
# should always show the mmproj file that will actually be used.
AUTO_PROJECTOR = "(auto)"

# IMAGE input slots, in transport order. The frontend reveals them on demand.
IMAGE_SLOTS = ("image", "image2", "image3", "image4", "image5", "image6", "image7", "image8")

JPEG_QUALITY = 90

# llama.cpp loads GGUF only, so only GGUF files are offered. Projector files are
# offered separately (see projector_choices).
LLM_WEIGHT_EXTENSIONS = {".gguf"}

# Split GGUF files only make sense for the language model. The first shard is
# the selectable one; llama.cpp opens the remaining shards itself.
_SPLIT_GGUF = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})(?P<ext>\.gguf)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedImage:
    """One image, resized/compressed for transport to the server."""

    slot: str
    data_base64: str
    digest: str
    source_size: tuple[int, int]
    sent_size: tuple[int, int]
    byte_size: int

    def data_url(self) -> str:
        return f"data:image/jpeg;base64,{self.data_base64}"


def ensure_llm_folder_registered() -> None:
    """Make sure the "LLM" model folder exists and accepts .gguf files."""
    folder_path = os.path.join(folder_paths.models_dir, LLM_FOLDER)
    extensions = set(getattr(folder_paths, "supported_pt_extensions", set())) | {".gguf"}
    try:
        if LLM_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[LLM_FOLDER] = ([folder_path], extensions)
        else:
            paths, registered = folder_paths.folder_names_and_paths[LLM_FOLDER]
            if folder_path not in paths:
                paths.append(folder_path)
            if isinstance(registered, set):
                registered.update(extensions)
            else:
                folder_paths.folder_names_and_paths[LLM_FOLDER] = (paths, set(registered) | extensions)
        os.makedirs(folder_path, exist_ok=True)
    except Exception as exc:  # Never block ComfyUI startup; the node reports it later.
        print(f"[LLM Session] Could not register the '{LLM_FOLDER}' model folder: {exc}", flush=True)


def _requested_folders() -> list[str]:
    requested = list(DEFAULT_MODEL_FOLDERS)
    extra = os.environ.get(MODEL_FOLDERS_ENV, "")
    requested.extend(part.strip() for part in extra.split(",") if part.strip())
    return requested


def model_roots() -> list[tuple[str, str]]:
    """(source, directory) pairs to scan, in priority order, duplicates removed."""
    ensure_llm_folder_registered()
    pairs: list[tuple[str, str]] = []
    seen_directories: set[str] = set()
    for entry in _requested_folders():
        if os.path.isabs(entry):
            directories = [entry]
        elif entry in folder_paths.folder_names_and_paths:
            try:
                directories = list(folder_paths.get_folder_paths(entry))
            except Exception as exc:
                print(f"[LLM Session] Could not resolve the '{entry}' model folder: {exc}", flush=True)
                directories = []
        else:
            print(
                f"[LLM Session] Ignoring '{entry}' in {MODEL_FOLDERS_ENV}: it is neither an existing "
                "ComfyUI model folder key nor an absolute path.",
                flush=True,
            )
            directories = []
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            key = os.path.realpath(directory)
            if key in seen_directories:
                continue
            seen_directories.add(key)
            pairs.append((entry, directory))
    return pairs


# Dropdown label -> absolute path, rebuilt with every listing so a name that
# exists in several scanned folders still resolves to the right file.
_choice_paths: dict[str, str] = {}
_logged_roots: tuple[str, ...] = ()


def scan_gguf_files() -> dict[str, str]:
    """Scan the configured folders for GGUF files: dropdown label -> path.

    Names are relative to their folder, so files in subfolders are offered as
    ``subfolder/name.gguf``. A file reachable through several folders is offered
    once (the first folder wins), and a name that would clash with a different
    file gets its folder key as a prefix.
    """
    global _choice_paths, _logged_roots
    pairs = model_roots()
    roots = tuple(f"{directory} ({source})" for source, directory in pairs)
    if roots and roots != _logged_roots:
        _logged_roots = roots
        print(f"[LLM Session] Scanning GGUF model folders: {', '.join(roots)}", flush=True)
    found: dict[str, str] = {}
    real_paths: set[str] = set()
    for source, root in pairs:
        for directory, dirnames, filenames in os.walk(root):
            relative_dir = "" if directory == root else os.path.relpath(directory, root)
            if relative_dir.count(os.sep) >= MODEL_SCAN_MAX_DEPTH:
                dirnames[:] = []
            dirnames.sort()
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() not in LLM_WEIGHT_EXTENSIONS:
                    continue
                path = os.path.join(directory, filename)
                real = os.path.realpath(path)
                if real in real_paths:
                    continue
                relative = relative_dir if relative_dir == "" else relative_dir.replace(os.sep, "/")
                name = f"{relative}/{filename}" if relative else filename
                if name in found:
                    name = f"{source}/{name}"
                found[name] = path
                real_paths.add(real)
    _choice_paths = found
    return found


def list_llm_files() -> list[str]:
    """All GGUF files found in the scanned folders (both models and projectors)."""
    return list(scan_gguf_files())


def _extension(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def _is_projector(name: str) -> bool:
    return "mmproj" in os.path.basename(name).lower()


def _is_later_split_shard(name: str) -> bool:
    match = _SPLIT_GGUF.match(os.path.basename(name))
    if match is None:
        return False
    try:
        index = int(match.group("index"))
        total = int(match.group("total"))
    except ValueError:
        return False
    return total >= 2 and index != 1


def model_choices() -> list[str]:
    """Selectable language models (projectors and extra shards excluded)."""
    return [
        name
        for name in list_llm_files()
        if not _is_projector(name)
        and _extension(name) in LLM_WEIGHT_EXTENSIONS
        and not _is_later_split_shard(name)
    ]


def projector_choices() -> list[str]:
    """Selectable vision projectors; the last entry means "no projector"."""
    projectors = [name for name in list_llm_files() if _is_projector(name) and _extension(name) == ".gguf"]
    return [*projectors, NO_PROJECTOR]


def projector_matches_model(projector_name: str, model_name: str) -> bool:
    """True if a projector is a plausible partner for a model, by name."""
    if projector_name in (NO_PROJECTOR, AUTO_PROJECTOR):
        return False
    return _score_match(_family_tokens(model_name), _family_tokens(projector_name)) > 0

# ---- name matching for the "(auto)" projector choice ----
#
# Ported from the frontend version that used to run when the model dropdown
# changed, so the choice stays honest at run time instead of at click time.

_QUANT_TAG_RE = re.compile(
    # Quantisation tags go first, as a whole: "q8_0", "q4_k_m", "iq4_xs",
    # "fp8_scaled", "int8_convrot", "bf16". Splitting them up afterwards would
    # leave stray digits behind and cause false matches.
    r"(?:^|[-_.])(?:iq[0-9]+|q[0-9]+|fp[0-9]+|bf16|f16|int[48]|nf[48])(?:[-_][a-z0-9]+)*"
)
_MMPROJ_TAG_RE = re.compile(r"(?:^|[-_.])mmproj(?=[-_.]|$)")
_SPLIT_SHARD_RE = re.compile(r"(?:^|[-_.])\d{5}-of-\d{5}$")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_SINGLE_LETTER_RE = re.compile(r"^[a-z]$")
_PRECISION_RE = re.compile(r"f16|bf16", re.IGNORECASE)
_Q8_RE = re.compile(r"q8", re.IGNORECASE)


def _family_tokens(name: str) -> list[str]:
    """Comparable name parts of a model or projector file name."""
    base = os.path.basename(str(name)).lower()
    if base.endswith(".gguf"):
        base = base[: -len(".gguf")]
    base = _QUANT_TAG_RE.sub(" ", base)
    base = _MMPROJ_TAG_RE.sub(" ", base)
    base = _SPLIT_SHARD_RE.sub(" ", base)
    # Keep digits (the "5" of Qwen2.5), drop single letters left over by tags.
    return [token for token in _TOKEN_SPLIT_RE.split(base) if token and not _SINGLE_LETTER_RE.match(token)]


def _score_match(model_tokens: list[str], projector_tokens: list[str]) -> float:
    if not model_tokens or not projector_tokens:
        return 0.0
    model_set = set(model_tokens)
    projector_set = set(projector_tokens)
    shared = len(model_set & projector_set)
    coverage = shared / len(model_set)
    if coverage < 0.5:
        return 0.0
    model_key = "-".join(model_tokens)
    projector_key = "-".join(projector_tokens)
    shares_prefix = 1.0 if model_key.startswith(projector_key) or projector_key.startswith(model_key) else 0.0
    return coverage + shared / len(projector_set) + shares_prefix


def _precision_rank(name: str) -> int:
    if _PRECISION_RE.search(name):
        return 0
    if _Q8_RE.search(name):
        return 1
    return 2


def rank_projectors(model_name: str, choices: list[str] | None = None) -> list[tuple[float, str]]:
    """Projectors that match a model, best first.

    On a score tie the higher-precision projector wins, then the alphabetically
    first one, so the order is stable across runs.
    """
    candidates = projector_choices() if choices is None else choices
    model_tokens = _family_tokens(model_name)
    if not model_tokens:
        return []
    ranked = []
    for choice in candidates:
        if choice in (NO_PROJECTOR, AUTO_PROJECTOR):
            continue
        score = _score_match(model_tokens, _family_tokens(choice))
        if score > 0:
            ranked.append((score, choice))
    ranked.sort(key=lambda entry: (-entry[0], _precision_rank(entry[1]), entry[1]))
    return ranked


def auto_match_projector(model_name: str, choices: list[str] | None = None) -> str | None:
    """Best-matching projector name for a model, or None if nothing matches."""
    ranked = rank_projectors(model_name, choices)
    return ranked[0][1] if ranked else None


def select_projector(model_name: str, current: str | None, text_only_pinned: bool, choices: list[str] | None = None) -> dict:
    """Decide which projector a run should use.

    Rules, in order:
      1. A projector already chosen by hand is kept if it is still on offer and its
         name fits the model as well as the best candidate does, so a deliberate
         choice (the Q8 instead of the F16 mmproj) survives re-runs and quant-only
         model changes - while a clearly worse one (a 27B mmproj on a 9B model) is
         replaced.
      2. '(none)' is kept only when the user picked it on purpose - that is how
         'run this VL model as text' persists across re-runs.
      3. Otherwise the best name match wins, and no projector if nothing matches.
    """
    choices = projector_choices() if choices is None else choices
    current = current or ""
    if current == NO_PROJECTOR and text_only_pinned:
        return {"selected": NO_PROJECTOR, "source": "kept (none)"}
    ranked = rank_projectors(model_name, choices)
    best_score = ranked[0][0] if ranked else 0.0
    available = [choice for choice in choices if choice not in (NO_PROJECTOR, AUTO_PROJECTOR)]
    if current in available and ranked:
        current_score = _score_match(_family_tokens(model_name), _family_tokens(current))
        if current_score >= best_score:
            return {"selected": current, "source": "kept existing"}
    if ranked:
        return {"selected": ranked[0][1], "source": "matched model"}
    return {"selected": NO_PROJECTOR, "source": "no match"}


def resolve_model_path(name: str, *, label: str) -> str | None:
    """Resolve a dropdown name to an existing file path (None for "no projector")."""
    if name == NO_PROJECTOR:
        return None
    if name == AUTO_PROJECTOR:
        raise ValueError(
            f"{label} is still '{AUTO_PROJECTOR}'. Pick a concrete file or report this: "
            "the node is meant to resolve it before starting the server."
        )
    scan_gguf_files()
    path = _choice_paths.get(name)
    if path is None:
        if os.path.isabs(name):
            path = name
        else:
            # Workflows saved before a folder was added may hold a bare name.
            path = next(
                (
                    candidate
                    for _source, directory in model_roots()
                    if os.path.isfile(candidate := os.path.join(directory, name))
                ),
                folder_paths.get_annotated_filepath(name),
            )
    if not os.path.isfile(path):
        scanned = ", ".join(directory for _source, directory in model_roots()) or "(none configured)"
        raise FileNotFoundError(
            f"{label} '{name}' was not found. This node scans these GGUF model folders: {scanned}. "
            f"Check that the file still exists, or pick it again from the list."
        )
    return path


def _to_pil(tensor, frame_index: int) -> Image.Image:
    if tensor is None:
        raise ValueError("Received no image data.")
    if hasattr(tensor, "shape") and len(tensor.shape) >= 3 and tensor.shape[0] > frame_index:
        frame = tensor[frame_index]
    elif len(getattr(tensor, "shape", ())) >= 3:
        raise ValueError(f"Requested image frame {frame_index + 1} is not present in the batch.")
    else:
        frame = tensor
    array = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.BICUBIC,
    )


def _encode_jpeg(image: Image.Image) -> bytes:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        image = background.convert("RGB")
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    except Exception as exc:
        print(f"[LLM Session] Optimized JPEG encoding failed, using standard JPEG: {exc}", flush=True)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def prepare_image(tensor, slot: str, frame_index: int, max_edge: int) -> PreparedImage:
    """Resize/compress one tensor frame and identify the prepared bytes.

    The digest identifies the *prepared* image, so a different slot or a
    different picture can never be mistaken for the same session input.
    """
    source = _to_pil(tensor, frame_index)
    prepared = _resize_to_max_edge(source, max_edge)
    data = _encode_jpeg(prepared)
    encoded = base64.b64encode(data).decode("ascii")
    return PreparedImage(
        slot=slot,
        data_base64=encoded,
        digest=hashlib.sha256(encoded.encode("ascii")).hexdigest(),
        source_size=source.size,
        sent_size=prepared.size,
        byte_size=len(data),
    )


# ---------------------------------------------------------------------------
# Minimal GGUF metadata reader, used only to check whether a model's chat
# template understands model-specific options such as "enable_thinking".
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"
_GGUF_UINT8, _GGUF_INT8, _GGUF_UINT16, _GGUF_INT16 = 0, 1, 2, 3
_GGUF_UINT32, _GGUF_INT32, _GGUF_FLOAT32, _GGUF_BOOL = 4, 5, 6, 7
_GGUF_STRING, _GGUF_ARRAY, _GGUF_UINT64, _GGUF_INT64, _GGUF_FLOAT64 = 8, 9, 10, 11, 12


def _read_string(handle) -> str:
    length = struct.unpack("<Q", handle.read(8))[0]
    if length > 64 * 1024 * 1024:
        raise ValueError("GGUF string is too large to read.")
    return handle.read(length).decode("utf-8", errors="replace")


_GGUF_SIZES = {
    _GGUF_UINT8: 1, _GGUF_INT8: 1, _GGUF_UINT16: 2, _GGUF_INT16: 2,
    _GGUF_UINT32: 4, _GGUF_INT32: 4, _GGUF_FLOAT32: 4, _GGUF_BOOL: 1,
    _GGUF_UINT64: 8, _GGUF_INT64: 8, _GGUF_FLOAT64: 8,
}
_GGUF_NUMBER_FORMATS = {
    _GGUF_UINT8: "<B", _GGUF_INT8: "<b", _GGUF_UINT16: "<H", _GGUF_INT16: "<h",
    _GGUF_UINT32: "<I", _GGUF_INT32: "<i", _GGUF_FLOAT32: "<f",
    _GGUF_UINT64: "<Q", _GGUF_INT64: "<q", _GGUF_FLOAT64: "<d",
}


def _read_value(handle, value_type: int, strings: bool, numbers: bool):
    """Read (or skip) one GGUF metadata value, keeping only the wanted kinds."""
    if value_type == _GGUF_STRING:
        text = _read_string(handle)
        return text if strings else None
    if value_type == _GGUF_ARRAY:
        element_type = struct.unpack("<I", handle.read(4))[0]
        count = struct.unpack("<Q", handle.read(8))[0]
        if count > 2_000_000:
            raise ValueError("GGUF array is too large to read.")
        for _ in range(count):
            _read_value(handle, element_type, strings, False)
        return None
    if value_type == _GGUF_BOOL:
        raw = handle.read(1)
        return bool(raw[0]) if numbers else None
    if value_type in _GGUF_NUMBER_FORMATS:
        raw = handle.read(_GGUF_SIZES[value_type])
        if numbers:
            return struct.unpack(_GGUF_NUMBER_FORMATS[value_type], raw)[0]
        return None
    raise ValueError(f"Unknown GGUF metadata type {value_type}")


def gguf_metadata(path: str, *, string_keys: tuple[str, ...] = (), number_key_parts: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Read selected string and numeric metadata entries from a GGUF header.

    Only the header is read, so this is cheap even for a 18 GB model file.
    """
    strings: dict[str, str] = {}
    numbers: dict[str, float] = {}
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return strings, numbers
            version = struct.unpack("<I", handle.read(4))[0]
            count_format = "<I" if version <= 1 else "<Q"
            size = 8 if version > 1 else 4
            struct.unpack(count_format, handle.read(size))  # tensor count
            key_count = struct.unpack(count_format, handle.read(size))[0]
            for _ in range(min(key_count, 40000)):
                key = _read_string(handle)
                value_type = struct.unpack("<I", handle.read(4))[0]
                want_string = key in string_keys
                want_number = bool(number_key_parts) and any(part in key for part in number_key_parts)
                value = _read_value(handle, value_type, want_string, want_number)
                if want_string and isinstance(value, str):
                    strings[key] = value
                if want_number and isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers[key] = value
            return strings, numbers
    except (OSError, ValueError, struct.error) as exc:
        print(f"[LLM Session] Could not read GGUF metadata from {os.path.basename(path)}: {exc}", flush=True)
    return strings, numbers


def template_supports_option(model_path: str, option: str) -> bool:
    """True when the model's own chat template mentions ``option``."""
    template = gguf_metadata(model_path, string_keys=("tokenizer.chat_template",))[0].get("tokenizer.chat_template")
    if not template:
        return False
    return re.search(rf"\b{re.escape(option)}\b", template) is not None


def mtp_predict_layers(model_path: str) -> int:
    """MTP / NextN prediction layers declared by the model (0 = speculative MTP unavailable)."""
    numbers = gguf_metadata(model_path, number_key_parts=("nextn_predict_layers", "mtp_predict_layers"))[1]
    if not numbers:
        return 0
    return int(max(numbers.values()))
