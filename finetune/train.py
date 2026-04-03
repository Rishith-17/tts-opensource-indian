"""
Finetune FastPitch TTS on IndicTTS Kannada data.

Usage:
    python finetune/train.py --lang kn --batch_size 8 --evaluate
    python finetune/train.py --lang kn --eval-only
    python finetune/train.py --lang kn --deploy-only

RTX 4050 (6GB VRAM): use --batch_size 8
Stop with Ctrl+C when eval loss plateaus (~20-40 epochs)
"""
import os, sys, json, argparse, shutil
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_paths(lang):
    return {
        "model_dir":  os.path.join(ROOT, "models", lang, "fastpitch"),
        "checkpoint": os.path.join(ROOT, "models", lang, "fastpitch", "best_model.pth"),
        "config":     os.path.join(ROOT, "models", lang, "fastpitch", "config.json"),
        "speakers":   os.path.join(ROOT, "models", lang, "fastpitch", "speakers.pth"),
        "data":       os.path.join(ROOT, "finetune", "data", lang),
        "metadata":   os.path.join(ROOT, "finetune", "data", lang, "metadata.csv"),
        "output":     os.path.join(ROOT, "finetune", "output", lang),
        "backup":     os.path.join(ROOT, "models", lang, "fastpitch_original"),
        "eval_dir":   os.path.join(ROOT, "finetune", "eval", lang),
    }

def validate(p, lang):
    ok = True
    if not os.path.exists(p["checkpoint"]):
        print(f"✗ Base model missing: {p['checkpoint']}")
        ok = False
    if not os.path.exists(p["metadata"]):
        print(f"✗ Dataset missing: {p['metadata']}")
        print(f"  Run: python finetune/prepare_dataset.py --lang {lang}")
        ok = False
    if not ok:
        sys.exit(1)
    print("✓ Paths OK")

def kn_formatter(root_path, meta_file, **kwargs):
    """Custom formatter: assigns 'female' speaker to match pretrained model."""
    import csv
    items = []
    with open(os.path.join(root_path, meta_file), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 2:
                continue
            items.append({
                "text":         row[1].strip(),
                "audio_file":   os.path.join(root_path, "wavs", row[0] + ".wav"),
                "speaker_name": "female",
                "root_path":    root_path,
            })
    return items

def build_config(p, args):
    from TTS.config import BaseDatasetConfig, load_config

    config = load_config(p["config"])
    speakers_file = p["speakers"] if os.path.exists(p["speakers"]) else None

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",          # placeholder — overridden by formatter= kwarg in load_tts_samples
        dataset_name=f"{args.lang}_ft",
        path=p["data"],
        meta_file_train="metadata.csv",
        meta_file_val=None,
        language=args.lang,
        ignored_speakers=None,
    )

    config.datasets            = [dataset_config]
    config.batch_size          = args.batch_size
    config.eval_batch_size     = max(4, args.batch_size // 2)
    config.num_loader_workers      = 4   # Linux handles multiprocessing well; use 0 on Windows
    config.num_eval_loader_workers = 2
    config.precompute_num_workers  = 0
    config.batch_size              = args.batch_size
    config.eval_batch_size         = max(4, args.batch_size // 2)

    # Keep pitch enabled — required to match pretrained checkpoint weights
    config.run_eval            = True
    config.epochs              = args.epochs
    config.save_step           = 500
    config.save_n_checkpoints  = 3
    config.save_best_after     = 500
    config.print_step          = 50
    config.plot_step           = 200
    config.lr                  = args.lr
    config.lr_scheduler        = "NoamLR"
    config.lr_scheduler_params = {"warmup_steps": 2000}
    config.grad_clip           = 5.0
    config.output_path         = p["output"]
    config.run_name            = f"{args.lang}_finetune"
    config.project_name        = "IndicTTS_Finetune"

    if speakers_file:
        config.speakers_file = os.path.abspath(speakers_file)
        if hasattr(config, "model_args") and config.model_args:
            config.model_args.speakers_file = os.path.abspath(speakers_file)

    return config

def deploy(p, lang):
    """Swap finetuned model into models/<lang>/fastpitch/."""
    import glob
    best = os.path.join(p["output"], "best_model.pth")
    if not os.path.exists(best):
        candidates = glob.glob(os.path.join(p["output"], "**", "*.pth"), recursive=True)
        if not candidates:
            print("✗ No checkpoint found to deploy")
            return
        candidates.sort(key=os.path.getmtime, reverse=True)
        best = candidates[0]
    if not os.path.exists(p["backup"]):
        shutil.copytree(p["model_dir"], p["backup"])
        print(f"✓ Original backed up → {p['backup']}")
    shutil.copy2(best, p["checkpoint"])
    print(f"✓ Deployed: {best} → {p['checkpoint']}")
    print(f"  Restart api_server.py to use the finetuned model.")

def _cer(ref, hyp):
    """Character Error Rate (edit distance / ref length)."""
    r, h = list(ref.strip()), list(hyp.strip())
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)

def evaluate(p, lang, n_samples=20):
    """Generate test samples and measure accuracy via Whisper CER."""
    import numpy as np
    import importlib.util

    print(f"\n{'='*50}")
    print(f"  Evaluating [{lang.upper()}] model")
    print(f"{'='*50}")

    # Load test texts from metadata
    test_texts = []
    with open(p["metadata"], encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                test_texts.append(parts[1])
    test_texts = test_texts[:n_samples]

    # Load TTS model
    spec = importlib.util.spec_from_file_location("tts_mod", os.path.join(ROOT, "tts.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tts = mod.MultilingualTTS()

    # Load Whisper for CER
    whisper_ok = False
    try:
        import whisper
        wm = whisper.load_model("small")
        whisper_ok = True
        print("  Whisper loaded for CER measurement")
    except Exception:
        print("  Whisper not available — will only generate audio")

    os.makedirs(p["eval_dir"], exist_ok=True)
    results = []

    for i, text in enumerate(test_texts):
        try:
            out = os.path.join(p["eval_dir"], f"eval_{i:03d}.wav")
            wav = tts.generate_speech(text, lang, "female")
            tts.save_audio(wav, out)
            row = {"text": text, "file": out}
            if whisper_ok:
                hyp = wm.transcribe(out, language=lang)["text"].strip()
                cer = _cer(text, hyp)
                row.update({"hypothesis": hyp, "cer": round(cer, 3)})
                print(f"  [{i+1:02d}/{n_samples}] CER={cer:.1%}  '{text[:45]}'")
            else:
                print(f"  [{i+1:02d}/{n_samples}] Saved: {out}")
            results.append(row)
        except Exception as e:
            print(f"  [{i+1:02d}] Error: {e}")

    # Print summary
    cer_vals = [r["cer"] for r in results if "cer" in r]
    if cer_vals:
        avg_cer  = np.mean(cer_vals)
        accuracy = (1 - avg_cer) * 100
        print(f"\n  ┌─────────────────────────────────┐")
        print(f"  │  Kannada TTS Accuracy Report    │")
        print(f"  ├─────────────────────────────────┤")
        print(f"  │  Samples tested : {len(results):<15} │")
        print(f"  │  Avg CER        : {avg_cer:.1%:<15} │")
        print(f"  │  Accuracy       : {accuracy:.1f}%{'':<13} │")
        print(f"  │  Audio files    : finetune/eval/ │")
        print(f"  └─────────────────────────────────┘")

    # Save JSON report
    report = os.path.join(p["eval_dir"], "results.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"lang": lang, "accuracy": round(accuracy if cer_vals else 0, 2),
                   "avg_cer": round(avg_cer if cer_vals else 0, 4),
                   "samples": results}, f, ensure_ascii=False, indent=2)
    print(f"  Full report: {report}")

# ── Main training ─────────────────────────────────────────────────────────────

def main(args):
    p = get_paths(args.lang)
    validate(p, args.lang)

    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    config = build_config(p, args)

    # Fix pitch_fmin=0 — must be done BEFORE AudioProcessor is created
    if hasattr(config, "audio") and hasattr(config.audio, "pitch_fmin"):
        if not config.audio.pitch_fmin or config.audio.pitch_fmin <= 0:
            config.audio.pitch_fmin = 65.0   # ~C2, safe floor for voice F0

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("Loading dataset...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets, eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.02,
        formatter=kn_formatter,
    )
    print(f"  Train: {len(train_samples)}  Eval: {len(eval_samples)}")

    # Build SpeakerManager from speakers.pth if model is multi-speaker
    from TTS.tts.utils.speakers import SpeakerManager
    speaker_manager = None
    if os.path.exists(p["speakers"]):
        speaker_manager = SpeakerManager(speaker_id_file_path=p["speakers"])
        print(f"✓ SpeakerManager loaded ({speaker_manager.num_speakers} speakers)")

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
    print(f"Loading pretrained weights: {p['checkpoint']}")
    model.load_checkpoint(config, p["checkpoint"], eval=False)
    # Move entire model + speaker embeddings to GPU
    model = model.cuda() if torch.cuda.is_available() else model
    print(f"✓ Pretrained weights loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Patch forward_tts to move speaker IDs to GPU before embedding lookup
    if torch.cuda.is_available():
        _orig_forward_encoder = model._forward_encoder
        def _patched_forward_encoder(x, x_mask, g=None):
            if g is not None and g.device.type == "cpu":
                g = g.cuda()
            return _orig_forward_encoder(x, x_mask, g)
        model._forward_encoder = _patched_forward_encoder

    # Use shared f0_cache — check if GPU precompute already ran
    f0_cache      = os.path.join(ROOT, "finetune", "cache", args.lang, "f0_cache")
    phoneme_cache = os.path.join(ROOT, "finetune", "cache", args.lang, "phoneme_cache")
    pitch_stats   = os.path.join(f0_cache, "pitch_stats.npy")
    os.makedirs(phoneme_cache, exist_ok=True)

    if not os.path.exists(pitch_stats):
        print("\n⚠ F0 cache incomplete. Run GPU precompute first:")
        print(f"  python finetune/precompute_f0_gpu.py --lang {args.lang}")
        print("Then re-run training.\n")
        # Still set the path — TTS will recompute on CPU as fallback
        if os.path.exists(f0_cache) and not os.path.exists(pitch_stats):
            shutil.rmtree(f0_cache)  # wipe incomplete so TTS rebuilds it

    if hasattr(config, "f0_cache_path"):
        config.f0_cache_path = f0_cache
    if hasattr(config, "phoneme_cache_path"):
        config.phoneme_cache_path = phoneme_cache
    if hasattr(config, "audio") and hasattr(config.audio, "stats_path"):
        config.audio.stats_path = None

    # Disable wandb/tensorboard loggers — use console only
    config.dashboard_logger = "tensorboard"
    try:
        import tensorboard  # noqa
    except ImportError:
        config.dashboard_logger = ""   # no dashboard, console only

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=1,
            use_accelerate=False,
        ),
        config, output_path=p["output"],
        model=model, train_samples=train_samples, eval_samples=eval_samples,
        parse_command_line_args=False,
    )

    print(f"\n{'='*50}")
    print(f"  Finetuning [{args.lang.upper()}]")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"  Output: {p['output']}")
    print(f"  Tip: Ctrl+C to stop early when eval loss plateaus")
    print(f"{'='*50}\n")

    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\nTraining stopped early.")

    if args.deploy or args.evaluate:
        deploy(p, args.lang)

    if args.evaluate:
        evaluate(p, args.lang)
    elif not args.deploy:
        print(f"\nTo deploy: python finetune/train.py --lang {args.lang} --deploy-only")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",        default="kn")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--deploy",      action="store_true", help="Deploy best model after training")
    parser.add_argument("--deploy-only", action="store_true", help="Just deploy, skip training")
    parser.add_argument("--evaluate",    action="store_true", help="Deploy + accuracy report after training")
    parser.add_argument("--eval-only",   action="store_true", help="Just run accuracy report")
    args = parser.parse_args()

    p = get_paths(args.lang)
    if args.deploy_only:
        deploy(p, args.lang)
    elif args.eval_only:
        evaluate(p, args.lang)
    else:
        main(args)
