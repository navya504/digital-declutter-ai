"""Merge a local PEFT LoRA adapter into its base model for Ollama import."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import warnings
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Folder containing adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B",
                        help="Exact base model used for the Nebius fine-tune")
    parser.add_argument("--base-output", type=Path, default=Path("artifacts/ollama-base"))
    parser.add_argument("--merged-output", type=Path, default=Path("artifacts/ollama-merged"))
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"Missing {adapter_dir / 'adapter_config.json'}")
    if not any(adapter_dir.glob("adapter_model*.safetensors")):
        raise SystemExit(f"No adapter_model*.safetensors found in {adapter_dir}")
    if args.base_output.resolve() == args.merged_output.resolve():
        raise SystemExit("--base-output and --merged-output must be different folders")

    try:
        from peft import PeftModel
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install merge dependencies with: pip install transformers peft accelerate safetensors") from exc

    def save_tokenizer(tokenizer, destination: Path) -> None:
        tokenizer.save_pretrained(destination)
        # Qwen2 GGUF conversion needs its GPT-2 BPE vocabulary files. Fast-tokenizer
        # saves may include tokenizer.json without vocab.json and merges.txt.
        if not (destination / "tokenizer.model").is_file() and not (
            (destination / "vocab.json").is_file() and (destination / "merges.txt").is_file()
        ):
            slow_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
            slow_tokenizer.save_pretrained(destination)

    def normalized_adapter_copy(source: Path, destination: Path) -> int:
        shutil.copytree(source, destination)
        remapped_count = 0
        weight_files = sorted(destination.glob("adapter_model*.safetensors"))
        if not weight_files:
            raise SystemExit(f"No adapter_model*.safetensors found in {destination}")
        for weight_file in weight_files:
            state = load_file(str(weight_file), device="cpu")
            normalized = {}
            key_map = {}
            for key, tensor in state.items():
                new_key = key
                # Nebius exports these keys relative to Qwen's `model`, while
                # current PEFT expects the wrapped `base_model.model` prefix.
                if key.startswith("model."):
                    new_key = "base_model.model." + key
                    remapped_count += 1
                if new_key in normalized:
                    raise SystemExit(f"Adapter key collision while normalizing {new_key}")
                normalized[new_key] = tensor.contiguous()
                key_map[key] = new_key
            save_file(normalized, str(weight_file), metadata={"format": "pt"})

            index_path = weight_file.with_name(weight_file.name + ".index.json")
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                index["weight_map"] = {key_map.get(k, k): v for k, v in index["weight_map"].items()}
                index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return remapped_count

    print(f"Loading the exact base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    args.base_output.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, args.base_output)
    base.save_pretrained(args.base_output, safe_serialization=True, max_shard_size="2GB")
    print(f"Saved base model for Ollama: {args.base_output.resolve()}")

    print(f"Loading adapter: {adapter_dir}")
    with tempfile.TemporaryDirectory(prefix="declutter-normalized-adapter-", dir=adapter_dir.parent) as temp_dir:
        normalized_dir = Path(temp_dir) / "adapter"
        remapped_count = normalized_adapter_copy(adapter_dir, normalized_dir)
        print(f"Normalized {remapped_count} Nebius adapter tensor keys for PEFT.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuned = PeftModel.from_pretrained(base, str(normalized_dir), is_trainable=False)
        missing_adapter_warnings = [
            str(w.message) for w in caught if "Found missing adapter keys" in str(w.message)
        ]
        if missing_adapter_warnings:
            raise SystemExit("Adapter weights did not all load; refusing to merge randomly initialized weights. " + missing_adapter_warnings[0])
    merged = tuned.merge_and_unload()
    args.merged_output.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, args.merged_output)
    merged.save_pretrained(args.merged_output, safe_serialization=True, max_shard_size="2GB")
    print(f"Saved merged fine-tuned model for Ollama: {args.merged_output.resolve()}")
    base_modelfile = Path("Modelfile.ollama-base")
    tuned_modelfile = Path("Modelfile.ollama-tuned")
    base_gguf = args.base_output.with_name(args.base_output.name + "-f16.gguf")
    tuned_gguf = args.merged_output.with_name(args.merged_output.name + "-f16.gguf")
    def ollama_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return "./" + resolved.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return resolved.as_posix()

    base_modelfile.write_text(f"FROM {ollama_path(base_gguf)}\nPARAMETER temperature 0\n", encoding="utf-8")
    tuned_modelfile.write_text(f"FROM {ollama_path(tuned_gguf)}\nPARAMETER temperature 0\n", encoding="utf-8")
    print("Convert both folders to GGUF first (Ollama Safetensors import may not support this architecture):")
    print(f"python tools/llama.cpp/convert_hf_to_gguf.py {args.base_output.as_posix()} --outfile {base_gguf.as_posix()} --outtype f16")
    print(f"python tools/llama.cpp/convert_hf_to_gguf.py {args.merged_output.as_posix()} --outfile {tuned_gguf.as_posix()} --outtype f16")
    print(f"Created {base_modelfile} and {tuned_modelfile}; after conversion, import them with `ollama create`.")
    print("The adapter keys were validated before merging; the GGUF models use the same tokenizer and base weights.")


if __name__ == "__main__":
    main()
