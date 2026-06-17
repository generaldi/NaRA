from transformers import AutoTokenizer, AutoModel
import torch
from typing import Optional
import os

from config import (
    MODEL_TYPE,
    FINETUNING_TYPE,
    TASK_TYPE,
    MODEL_PATHS_MAPPING,
    CKPT_MAPPING,
    get_type,
)


def _load_nara(base_model, ckpt_path: str):
    """Load a NARA adapter checkpoint on top of `base_model`."""
    from nara import PeftModel, NARAModel

    print(f"[Info] Loading NARA model from: {ckpt_path}")
    model: NARAModel = PeftModel.from_pretrained(
        base_model,
        ckpt_path,
        torch_dtype=torch.bfloat16,
    )
    return model


def get_eval_model(
    base_model_name: str,
    peft_name: Optional[str] = None,
    ft_task: Optional[str] = None,
    run_time: int = 1,
    *args,
    **kwargs,
):
    """Load the base model (and optionally a NARA adapter) for evaluation.

    Simplified release version: only NARA is supported. The checkpoint is
    resolved in one of two ways:

    1. If `peft_name` is an existing directory, it is treated as a direct path
       to a NARA checkpoint folder and loaded as-is.
    2. Otherwise `peft_name` is interpreted as a FINETUNING_TYPE key (e.g.
       ``'nara'``) and the checkpoint path is looked up in
       ``CKPT_MAPPING[(model, task, NARA)][run_time - 1]``.

    Returns:
        (model, tokenizer, finetuning_type)
        `finetuning_type` is ``None`` when no adapter is loaded.
    """
    # Extra positional/keyword args (f_form, lr, embedding_dim, ablation_*, ...)
    # are accepted for call-site compatibility but unused in this NARA-only release.
    base_model_type: MODEL_TYPE = get_type(MODEL_TYPE, base_model_name)
    if base_model_type not in MODEL_PATHS_MAPPING:
        raise NotImplementedError(
            f"No path found in MODEL_PATHS_MAPPING for {base_model_type}, please specify one."
        )

    base_model = AutoModel.from_pretrained(
        MODEL_PATHS_MAPPING[base_model_type],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS_MAPPING[base_model_type], trust_remote_code=True
    )

    # No adapter requested: evaluate the base model directly.
    if not peft_name:
        return base_model, tokenizer, None

    # Case 1: peft_name is a direct path to a checkpoint directory.
    if os.path.isdir(peft_name):
        model = _load_nara(base_model, peft_name)
        return model, tokenizer, FINETUNING_TYPE.NARA

    # Case 2: resolve via CKPT_MAPPING.
    finetuning_type: FINETUNING_TYPE = get_type(FINETUNING_TYPE, peft_name)
    if finetuning_type != FINETUNING_TYPE.NARA:
        raise NotImplementedError(
            f"This release only supports NARA evaluation, got peft_name='{peft_name}'."
        )

    task_type: TASK_TYPE = get_type(TASK_TYPE, ft_task)
    key = (base_model_type, task_type, finetuning_type)
    if key not in CKPT_MAPPING:
        raise NotImplementedError(
            f"No entry found in CKPT_MAPPING for {key}. "
            f"Add the checkpoint path in config/format_input.py, "
            f"or pass peft_name=<path-to-checkpoint-dir> directly."
        )

    paths = CKPT_MAPPING[key]
    if not (1 <= run_time <= len(paths)) or not paths[run_time - 1]:
        raise ValueError(
            f"No checkpoint path for run_time={run_time} under {key}. "
            f"Fill in CKPT_MAPPING in config/format_input.py."
        )

    model = _load_nara(base_model, paths[run_time - 1])
    return model, tokenizer, finetuning_type
