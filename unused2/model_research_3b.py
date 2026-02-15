#!/usr/bin/env python3
"""Qwen2.5-VL-3B-Instruct optimized for 12GB GPU with float16.

The 3B model in float16 is ~6GB, leaving plenty of room for inference on 12GB GPU.
This is the proven working configuration.
"""

import argparse
import gc
import os
from io import BytesIO

import torch
from PIL import Image


def load_image(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests

        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if path_or_url.startswith("file://"):
        path_or_url = path_or_url[7:]
    return Image.open(path_or_url).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 3B optimized for 12GB GPU")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", required=True, help="Local file path or URL")
    parser.add_argument("--prompt", default="Describe exactly what is visible. Avoid guessing; if uncertain, say 'unclear'.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--resized-width", type=int, default=336)
    parser.add_argument("--resized-height", type=int, default=336)
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()

    # Clear GPU before loading
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory before load: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    # Load model with float16 (proven working config)
    print("Loading 3B model with float16...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
    print(f"Model loaded. Device: {model.device}")

    # Load and process image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "resized_width": args.resized_width, "resized_height": args.resized_height},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    inputs = inputs.to("cuda")

    # Generate
    print("Generating...")
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print("\n" + "=" * 60)
    print("OUTPUT:")
    print("=" * 60)
    print(output_text[0])
    print("=" * 60)


if __name__ == "__main__":
    main()
