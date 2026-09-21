# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native encoding interface doubles; no tokenizer, checkpoint or inference.

Inspected at ComfyUI c194dd00cd42aa18d9dbf27d977bf6b85d9ea565:
comfy/text_encoders/krea2.py, comfy/text_encoders/qwen3vl.py and comfy/sd.py.
"""

import sys
from types import ModuleType

import torch

KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


class Krea2Tokenizer:
    pass


class Krea2TEModel:
    pass


def install(monkeypatch):
    for name, attributes in (
        ("comfy.text_encoders", {}),
        (
            "comfy.text_encoders.krea2",
            {
                "KREA2_TEMPLATE": KREA2_TEMPLATE,
                "Krea2Tokenizer": Krea2Tokenizer,
                "Krea2TEModel": Krea2TEModel,
            },
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        parent, key = name.rsplit(".", 1)
        monkeypatch.setattr(sys.modules[parent], key, module, raising=False)


class Clip:
    def __init__(self):
        self.tokenizer = Krea2Tokenizer()
        # Native te() constructs a subclass for dtype/quantization options.
        self.cond_stage_model = type("Krea2TEModel_", (Krea2TEModel,), {})()
        self.tokenized = []
        self.encoded = []
        self.outputs = []
        self.schedules = [{}]
        self.dtype = torch.float32

    def tokenize(self, text, *, images=(), llama_template=None):
        # Deliberately model the generic image-template default, so omitting
        # the explicit native Krea2 template changes the recorded user turn.
        template = llama_template or (
            "generic image template: {}" if images else KREA2_TEMPLATE
        )
        tokens = {
            "text": text,
            "images": images,
            "template": template,
            "rendered": template.format(text),
        }
        self.tokenized.append(tokens)
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        images = tokens["images"]
        value = images[0].mean().item() * 10 + len(tokens["text"]) if images else -2
        token_count = len(tokens["text"]) % 3 + 1
        basis = torch.arange(30720, dtype=torch.float32).reshape(1, 1, -1) / 30720
        entries = []
        for index, schedule in enumerate(self.schedules):
            embedding = (
                (basis + value + index).expand(1, token_count, -1).to(self.dtype)
            )
            entries.append(
                [
                    embedding,
                    {
                        "pooled_output": torch.tensor([[value]]),
                        "attention_mask": torch.ones((1, token_count)),
                        **schedule,
                    },
                ]
            )
        self.outputs.append(entries)
        return entries
