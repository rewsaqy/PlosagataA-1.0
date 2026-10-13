"""Fine-tune LoRA chat 0.5B/1B di DEVICE LAIN (GPU). Jangan dijalankan di CPU sini.

Di device GPU (Colab/Kaggle/PC RTX), jalankan:
  pip install transformers peft trl datasets accelerate bitsandbytes
  python train_chat_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --data data/contoh_chat.jsonl --out lora-hoax-chat

Untuk 1B pas, ganti --base ke meta-llama/Llama-3.2-1B-Instruct (butuh `huggingface-cli login`).
Hasil: folder lora-hoax-chat/ berisi adapter. Copy kembali ke sini,
lalu jalan: python chat.py --model <base>  (adapter dimuat otomatis jika ada, lihat chat_lora_infer.py)
"""
import argparse
import json


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/contoh_chat.jsonl")
    ap.add_argument("--out", default="lora-hoax-chat")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        print("[WARN] Tidak ada GPU di device ini. Pindahkan script ini ke device GPU (Colab/Kaggle). Berhenti.")
        raise SystemExit(1)

    rows = load_rows(a.data)
    print(f"Data chat: {len(rows)} contoh")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(ex):
        return tok.apply_chat_template(ex["messages"], tokenize=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        a.base, load_in_4bit=True, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=a.lr, fp16=True, logging_steps=10, save_steps=100,
        save_total_limit=2, report_to="none",
    )
    SFTTrainer(model=model, train_dataset=ds, dataset_text_field="text",
               args=args, tokenizer=tok).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Selesai. Adapter di {a.out}/ — copy folder ini ke device inferensi.")


if __name__ == "__main__":
    main()
