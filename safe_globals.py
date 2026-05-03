"""
FILE: safe_globals.py
ROLE: Centralized PyTorch Safe Globals Registry
-------------------------------------------------------------------------
DESCRIPTION:
Registers all trusted classes required for torch.load(weights_only=True)
across the pipeline. Self-healing mode catches unknown blocked globals
at runtime and registers them automatically, then retries the load.

COVERS:
- Ultralytics / YOLO model classes
- OmegaConf (required by Pyannote VAD)
- typing module (required by Pyannote VAD)
- Core PyTorch & stdlib types
-------------------------------------------------------------------------
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import re
import typing
import collections
import inspect
import torch
import warnings
warnings.filterwarnings("ignore", message=".*pyannote.audio 0\\.0\\.1.*")
warnings.filterwarnings("ignore", message=".*torch 1\\.10\\.0.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*std.*degrees of freedom.*")

# ── Methods ────────────────────────────────────────────────────────────────────

def patch_hf_hub():
    """
    Patches huggingface_hub.hf_hub_download across all modules that hold
    a direct reference to it. Required because Pyannote and WhisperX still
    pass 'use_auth_token' which newer HF hub versions dropped in favour of 'token'.
    
    Call once at startup alongside register_omegaconf_only() or register_all().
    """
    import huggingface_hub
    import huggingface_hub.file_download
    import huggingface_hub.utils
    import pyannote.audio.core.pipeline
    import pyannote.audio.core.model
    import pyannote.audio.pipelines.utils.getter
    import whisperx.diarize
    from huggingface_hub import hf_hub_download as real_download

    def patched_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return real_download(*args, **kwargs)

    _targets = [
        huggingface_hub,
        huggingface_hub.file_download,
        huggingface_hub.utils,
        pyannote.audio.core.pipeline,
        pyannote.audio.core.model,
        pyannote.audio.pipelines.utils.getter,
        whisperx.diarize,
    ]
    for module in _targets:
        if hasattr(module, "hf_hub_download"):
            setattr(module, "hf_hub_download", patched_download)

    print("[safe_globals] ✅ Patched hf_hub_download across all pyannote/whisperx modules")


def _get_classes(module):
    """Return all classes defined directly in the given module."""
    return [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if module.__name__ in (obj.__module__ or "")
    ]


def _get_omegaconf_classes():
    import omegaconf.listconfig
    import omegaconf.dictconfig
    import omegaconf.base
    return [
        *_get_classes(omegaconf.listconfig),
        *_get_classes(omegaconf.dictconfig),
        *_get_classes(omegaconf.base),
        *_get_classes(omegaconf.nodes),
    ]


def _get_typing_globals():
    """Register all public members of the typing module."""
    return [
        obj for obj in vars(typing).values()
        if callable(obj) or isinstance(obj, type)
    ]


def _get_ultralytics_classes():
    import ultralytics.nn.modules.conv as ulconv
    import ultralytics.nn.modules.block as ulblock
    import ultralytics.nn.modules.head as ulhead
    import ultralytics.utils as ulutils
    import ultralytics.utils.loss as ulloss
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel
    )
    return [
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel,
        *_get_classes(ulutils),
        *_get_classes(ulloss),
        *_get_classes(ulconv),
        *_get_classes(ulblock),
        *_get_classes(ulhead),
    ]


def _get_pyannote_classes():
    """
    Register all pyannote.audio classes required by Pyannote checkpoints.
    Covers Specifications and any other classes serialized into the checkpoint.
    """
    import pyannote.audio.core.task
    import pyannote.audio.core.model
    import pyannote.audio.pipelines.speaker_diarization as spk
    import pyannote.audio.pipelines.speaker_verification as spv

    return [
        *_get_classes(pyannote.audio.core.task),
        *_get_classes(pyannote.audio.core.model),
        *_get_classes(spk),
        *_get_classes(spv),
    ]

def _get_torch_stdlib_classes():
    return [
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.container.Sequential,
        collections.OrderedDict,
        torch.torch_version.TorchVersion,
    ]


def _get_builtin_types():
    """
    Register Python builtin types required by Pyannote/Lightning checkpoints.
    Covers list, dict, tuple, set, and other common builtins that PyTorch 2.6+
    no longer allows by default.
    """
    import builtins
    return [
        obj for obj in vars(builtins).values()
        if isinstance(obj, type)
    ]


def _resolve_global(module_name: str, class_name: str):
    """
    Dynamically import and return a class by module + name.
    Used by the self-healing loader to register unknown blocked globals.
    """
    import importlib
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name, None)
    except Exception:
        return None


def safe_torch_load(load_fn, *args, **kwargs):
    """
    Self-healing wrapper around any torch.load call.
    Catches WeightsUnpickler errors, extracts the blocked class,
    registers it, and retries — up to 10 times to handle checkpoints
    with multiple unknown globals.

    Usage:
        result = safe_torch_load(torch.load, path, map_location="cpu")
    """
    pattern = re.compile(r"GLOBAL (\S+)\.(\S+) was not an allowed global")
    
    for attempt in range(10):
        try:
            return load_fn(*args, **kwargs)
        except Exception as e:
            match = pattern.search(str(e))
            if not match:
                raise  # Not a safe-globals error — re-raise immediately
            
            module_name, class_name = match.group(1), match.group(2)
            resolved = _resolve_global(module_name, class_name)
            
            if resolved is None:
                raise RuntimeError(
                    f"Self-healing failed: could not import {module_name}.{class_name}"
                ) from e
            
            torch.serialization.add_safe_globals([resolved])
            print(f"[safe_globals] ⚡ Auto-registered: {module_name}.{class_name} (attempt {attempt + 1})")

    raise RuntimeError("Self-healing exceeded max retries (10). Check checkpoint integrity.")


def _patch_lightning_loader():
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
        import pytorch_lightning.utilities.cloud_io as pl_cloud_io

        def _patched(path, map_location=None, **kwargs):
            import torch, io
            if hasattr(path, "read"):
                data = path.read()
                return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)
            return torch.load(path, map_location=map_location, weights_only=False)

        # Patch both lightning_fabric AND pytorch_lightning
        cloud_io._load = _patched
        try:
            pl_cloud_io._load = _patched
        except Exception:
            pass  # pytorch_lightning may not be installed

        print("[safe_globals] ✅ Patched lightning_fabric._load → weights_only=False")

    except ImportError:
        pass


def register_omegaconf_only():
    torch.serialization.add_safe_globals([
        *_get_omegaconf_classes(),
        *_get_typing_globals(),
        *_get_builtin_types(),
        *_get_torch_stdlib_classes(),
        *_get_pyannote_classes(),
        collections.OrderedDict,
    ])
    _patch_lightning_loader()


def register_all():
    torch.serialization.add_safe_globals([
        *_get_omegaconf_classes(),
        *_get_typing_globals(),
        *_get_builtin_types(),
        *_get_ultralytics_classes(),
        *_get_pyannote_classes(),
        *_get_torch_stdlib_classes(),
    ])
    _patch_lightning_loader()