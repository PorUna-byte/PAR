# PAR — Reward Shaping to Mitigate Reward Hacking in RLHF

Reference implementation for **PAR**, a bounded, reference-centered reward
shaping function for RLHF, together with the full experimental harness used to
compare it against the standard reward-shaping baselines.

In RLHF, the policy is optimized against a *proxy* reward model. Because the
proxy is imperfect and unbounded, the policy eventually finds directions that
raise the proxy score without improving real response quality — reward hacking.
PAR attacks this by reshaping the reward instead of changing the RL algorithm:
for each prompt, the raw reward of a sampled response is compared against the
rewards of `K` reference responses drawn from the SFT policy for the *same*
prompt, and the shaped reward is the mean of the sigmoid margins,

```
r_shaped(x, y) = (1 / K) · Σ_k  σ( r(x, y) − r(x, y_ref_k) )
```

which is bounded in `(0, 1)`, centered per prompt, and saturates once the policy
is clearly ahead of its own reference distribution — removing the unbounded
gradient that drives hacking. See [`src/trainers/reward_shaper.py`](src/trainers/reward_shaper.py)
for PAR and every baseline in one place.

## What is in this repository

| | |
|---|---|
| **Base models** | `gemma2-2b`, `gemma2-9b`, `qwen3-4b-base`, `llama-3.1-8b` |
| **Datasets** | `ultrafb_bin` (UltraFeedback-Binarized), `hh_rlhf` (Anthropic HH-RLHF, helpful-base) |
| **Training objectives** | `sft`, `reward`, `reward_odin`, `ppo`, `a2c`, `grpo`, `dpo` |
| **Reward shaping** | `par`, `vanilla`, `warm`, `meanstd`, `clip`, `minmax`, `lsc`, `tanh`, `fittedpoly`, `sigmoid`, `sigmoidk2`, `sigmoidk3` |
| **Distributed backend** | DeepSpeed ZeRO + `torchrun`, optional vLLM/Ray rollout workers |
| **Evaluation** | LLM-as-a-judge pairwise winrate against the SFT reference (default judge `gpt-5-nano`), with token and USD cost accounting |

Everything runs from one entry point, [`src/train.py`](src/train.py), driven by
the argument surface in [`src/configs/config.py`](src/configs/config.py). The
Slurm pipelines in [`src/sbatch/`](src/sbatch/) are thin, resumable wrappers
around it.

---

## 1. Repository layout

```text
par/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── download_datasets.py         # download + paper-style preprocessing
│   └── dataset_summary.json         # row counts produced by the above
├── curve_package/
│   ├── json/                        # winrate/reward curves behind the paper figures
│   ├── figures/final/               # rendered PDF/PNG figures
│   └── scripts/draw_paper.py        # redraws figures from json/
└── src/
    ├── train.py                     # single entry point for every objective
    ├── combine_plot.py              # overlay rating curves across runs
    ├── configs/
    │   ├── config.py                # CLI surface, model/dataset/loss presets
    │   └── stage{2,3}_config.json   # DeepSpeed ZeRO configs
    ├── dataloaders/
    │   ├── dataset.py               # split paths, prompt formatting, HH/UF parsing
    │   └── dataloader.py            # SFT / PairedPreference / Prompt loaders
    ├── models/
    │   ├── models.py                # causal LM + scalar reward head (and ODIN head)
    │   └── loader.py                # builds policy/reference/reward/critic engines
    ├── trainers/
    │   ├── basic_trainer.py         # shared train/eval/sample/checkpoint loop
    │   ├── unpaired_trainer.py      # SFTTrainer
    │   ├── paired_trainer.py        # RewardTrainer, RewardTrainerODIN, DPOTrainer
    │   ├── ppo_trainer.py           # PPOTrainer
    │   ├── a2c_trainer.py           # A2CTrainer
    │   ├── grpo_trainer.py          # GRPOTrainer
    │   └── reward_shaper.py         # PAR + all reward-shaping baselines
    ├── rollouts/vllm_ray.py         # optional vLLM-on-Ray generation workers
    ├── merge/merge_rm_checkpoints.py# WARM: average reward-model checkpoints
    ├── llm_rating/
    │   ├── main.py                  # rate a run, write summary + curve
    │   ├── llm_comparator.py        # judge prompt and API client
    │   └── pricing.py               # per-model $/1M-token table
    ├── scripts/
    │   ├── download_gemma2_2b.py    # one script per base model
    │   ├── download_gemma2_9b.py
    │   ├── download_qwen3_4b_base.py
    │   ├── download_llama_3_1_8b.py
    │   ├── generate_reference_responses.py   # PAR reference responses + rewards
    │   ├── merge_sample_outputs.py           # per-rank samples -> merged.json
    │   ├── autotune_batch_size.py            # OOM-safe batch-size probing
    │   ├── plot_preference_calibration.py
    │   └── plot_principle2_histogram.py
    ├── utils/
    │   ├── secret.py.example        # credential template (see §3)
    │   ├── utils.py                 # distributed helpers, logging
    │   ├── prompt.py, wandb_utils.py, reward_shaping_config.py
    └── sbatch/
        ├── _common.sh               # GPU detection, batch tables, resumable steps
        ├── 00_download_models.sh
        ├── 01_supervised_pipeline.sh
        ├── 02_rl_matrix_part1.sh    # PPO + A2C
        ├── 03_rl_matrix_part2.sh    # GRPO + DPO
        ├── 04_analysis.sh           # ablation suite
        └── 05_rating.sh             # LLM-judge winrate evaluation
```

Directories created at runtime and **not** tracked in git: `models_ck/`
(base weights), `data/<dataset>/` (preprocessed splits), `exp_runs/`
(checkpoints, samples, ratings), `pipeline_state/` (resume logs), `logs/`,
`results/`, `wandb/`.

---

## 2. Environment

```bash
conda create -n par python=3.13 -y
conda activate par
pip install -r requirements.txt
```

The pinned versions in [`requirements.txt`](requirements.txt) are the ones the
experiments were run with (PyTorch 2.10 + CUDA 12.8, Transformers 5.6,
DeepSpeed 0.18). `vllm` and `ray` are only needed if you use
`--rollout_backend vllm_ray` or `--generation_backend vllm`; the DeepSpeed
generation path works without them.

Hardware used per job: **one node, 4 × H200 (80GB+)**. Smaller GPUs work for
`gemma2-2b` if you lower the per-GPU batch sizes (see §8).

All Python entry points expect `src/` on the path. The Slurm helpers do this
for you; when running by hand:

```bash
export PROJECT_ROOT=/path/to/par
export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH
```

---

## 3. Credentials

Copy the template and export the tokens you need. `src/utils/secret.py` is
git-ignored — it reads everything from the environment, so never hardcode a key
in it:

```bash
cp src/utils/secret.py.example src/utils/secret.py

export HF_TOKEN=...          # required: Gemma and Llama are gated on the HF Hub
export OPENAI_API_KEY=...    # required for LLM-judge winrate evaluation
export WANDB_API_KEY=...     # optional; or use --wandb_mode offline / disabled
```

W&B logging is on by default (`--wandb_enabled true`, project `par`). To turn it
off entirely, pass `--wandb_enabled false`.

---

## 4. Data preparation

```bash
python data/download_datasets.py
```

Downloads `HuggingFaceH4/ultrafeedback_binarized` and
`Anthropic/hh-rlhf` (`helpful-base`) and applies the paper preprocessing:
prompt/chosen/rejected each truncated to `< 512` Gemma-2 tokens, UltraFeedback
rows kept only when `score_chosen > score_rejected` and neither response
mentions "confidence", HH prompts de-duplicated, test splits capped at 256
prompts, seed `22`.

Result:

```text
data/ultrafb_bin/{train_prefs,train_sft,test_prefs}.json
data/hh-rlhf-helpful/{train_prefs,train_sft,test_prefs}.json
data/dataset_summary.json
```

Useful flags: `--output_root`, `--tokenizer_name`, `--seed`.

## 5. Model download

```bash
python src/scripts/download_gemma2_2b.py
python src/scripts/download_gemma2_9b.py
python src/scripts/download_qwen3_4b_base.py
python src/scripts/download_llama_3_1_8b.py
```

or all four as one Slurm job (`MODELS=` selects a subset):

```bash
mkdir -p logs
sbatch src/sbatch/00_download_models.sh
```

Each script writes to the path the presets in `src/configs/config.py` expect:

```text
models_ck/gemma-2-2b/   models_ck/gemma-2-9b/
models_ck/qwen3-4b-base/   models_ck/llama-3.1-8b/
```

Roughly 38 GB in total. Accept the Gemma and Llama licences on the Hub first,
and make sure `HF_TOKEN` is exported.

---

## 6. Running the pipelines

Submit from the project root so that relative `logs/` paths resolve:

```bash
cd /path/to/par
mkdir -p logs
sbatch src/sbatch/01_supervised_pipeline.sh
```

Before submitting, adjust the Slurm header of each script to your cluster —
`--partition`, `--gres=gpu:h200:4`, `--time`, and the commented-out
`--nodelist`. The scripts activate a conda env named `par`; override with
`CONDA_ENV`, and pass `PROJECT_ROOT` if you submit from elsewhere:

```bash
sbatch --export=ALL,PROJECT_ROOT=/path/to/par,CONDA_ENV=myenv src/sbatch/01_supervised_pipeline.sh
```

**Every pipeline is resumable.** Each completed step appends a line to
`pipeline_state/<pipeline_name>/completed.log`, and a re-submitted job prints
`[SKIP]` for those steps. To force a full re-run, submit with
`RESET_PIPELINE_STATE=1`.

### Step 1 — supervised stage

```bash
sbatch src/sbatch/01_supervised_pipeline.sh
```

For all 4 models × 2 datasets, in order:

1. **SFT** — 2 epochs, lr `5e-6` → `exp_runs/sft_<model>_<dataset>/final_hf`
2. **Reward model** — 1 epoch, lr `5e-6`, `--save_every_eval true` with 5
   evaluation splits, so 5 intermediate checkpoints survive →
   `exp_runs/reward_<model>_<dataset>/`
3. **WARM merge** — averages those 5 reward checkpoints
   ([`merge_rm_checkpoints.py`](src/merge/merge_rm_checkpoints.py)) into
   `exp_runs/reward_<model>_<dataset>_warm/final_hf`, then deletes the
   intermediates (keep them with `--keep_step_checkpoints`)
4. **Reference responses** — samples `--num_refs 5` SFT responses per prompt and
   scores them with the reward model →
   `data/<dataset_dir>/<model>_{train_prefs,test_prefs}.json`, each row carrying
   `ref_responses` and `ref_rewards`. This is what PAR and LSC consume.

Generation for step 4 defaults to vLLM (`REFERENCE_GENERATION_BACKEND=vllm`);
set it to `hf` to use plain HuggingFace generation under `torchrun` instead.

### Step 2 — the RL matrix

```bash
sbatch src/sbatch/02_rl_matrix_part1.sh   # PPO, A2C  × {vanilla, par}
sbatch src/sbatch/03_rl_matrix_part2.sh   # GRPO, DPO × {vanilla, par}
```

Both iterate over all 4 models × 2 datasets. The policy and reference are
initialized from the matching SFT checkpoint, the reward and critic from the
matching reward model (or its `_warm` variant when `--reward_shaping warm`).
DPO is run with `vanilla` only, since it never queries a shaped reward during
training. Per-model/dataset learning rates, KL coefficients, epoch fractions
and buffer sizes live in the `_HP_*` tables at the top of each script and are
individually overridable by environment variable, e.g.
`GEMMA2_2B_RL_LR=8e-7 sbatch src/sbatch/02_rl_matrix_part1.sh`.

To widen the sweep, edit the arrays at the top of the script:

```bash
MODELS=(gemma2-2b gemma2-9b qwen3-4b-base llama-3.1-8b)
DATASETS=(hh_rlhf ultrafb_bin)
ALGOS=(ppo a2c)
SHAPINGS=(vanilla par)          # add warm meanstd clip minmax lsc ...
```

Each run evaluates and samples the test set 10 times per epoch
(`--eval_splits_per_epoch 10`), saves no policy checkpoints
(`--disable_checkpoint_saving true`), and finishes by merging per-rank samples
into `merged.json` for the judge.

### Step 3 — analysis / ablation suite

```bash
sbatch src/sbatch/04_analysis.sh
```

All ablations use one configuration — `ppo` + `gemma2-2b` + `hh_rlhf` (change
`BASE_MODEL` / `BASE_DATASET` at the top) — and cover:

| Run-name prefix | What it varies |
|---|---|
| `analysis_reward_{odin,reg}` | two extra reward models: ODIN (`--loss_name reward_odin`) and Reg (`--reward_reg`) |
| `analysis_suite_*` | the shaping baselines: Vanilla, WARM, ODIN, Reg, Meanstd, Clip, Minmax, LSC, PAR |
| `analysis_principle1_*` | boundedness: `--KL_coef {0.01, 0.05, 0.1}` and `--reward_ceil {5, 4, 3}` |
| `analysis_principle2_*` | shape: `tanh`, `fittedpoly`, `sigmoid`, `sigmoidk2`, `sigmoidk3`, each centered and uncentered |
| `analysis_dataeffi_num*` | data efficiency: `--num_refs {1, 3, 5}` references for PAR |
| `analysis_robust_*` | robustness: 2-epoch runs for PAR, LSC and Minmax |

Once ratings exist, re-submit with `RUN_ANALYSIS_CALIBRATION=1` to draw the
preference-calibration figure into `results/`.

### Step 4 — LLM-judge winrate evaluation

`conda activate par` before submitting so the compute node inherits the env
(this script does not activate one itself):

```bash
sbatch src/sbatch/05_rating.sh                                   # every sampled run
sbatch src/sbatch/05_rating.sh --run ppo_gemma2-2b_hh_rlhf_par   # one run
```

or directly, without Slurm:

```bash
python src/llm_rating/main.py \
  --run_dir exp_runs/ppo_gemma2-2b_hh_rlhf_par \
  --parallel_workers 16
```

For every `sample_on_test/step_*/merged.json`, each policy response is compared
against the SFT reference response for the same prompt. To cancel the judge's
position bias, each pair is judged **twice** with the presentation order
swapped; the two verdicts are then combined by the `FINAL_LABEL_MAP` in
[`llm_comparator.py`](src/llm_rating/llm_comparator.py) into a win (`1.0`), a
loss (`0.0`) or a tie (`0.5`) — a pair the two passes disagree on lands as a
tie, and an unparseable verdict is dropped from the average rather than counted.
Outputs per run:

```text
exp_runs/<run>/sample_on_test/step_*/merged_rated.json   # per-example verdicts
exp_runs/<run>/llm_rating_summary.json                   # winrate + tokens + USD per step
exp_runs/<run>/llm_rating_curve.png                      # winrate vs. training step
```

Rating is incremental — already-rated steps are skipped unless you pass
`--force` (`RATING_FORCE=1` for the Slurm wrapper). The default judge is
`gpt-5-nano` at low reasoning effort; costs are estimated from
[`src/llm_rating/pricing.py`](src/llm_rating/pricing.py), which you should
update if provider prices change.

### Step 5 — figures

```bash
# overlay the shaping variants of one run family
python src/combine_plot.py --prefix ppo_gemma2-2b_ultrafb_bin
python src/combine_plot.py --prefix analysis_principle2

# redraw the paper figures from the bundled curve data
python curve_package/scripts/draw_paper.py
```

`combine_plot.py` writes `exp_runs/<prefix>_combined_curve.png`;
`draw_paper.py` reads `curve_package/json/*.json` and writes PDFs and PNGs to
`curve_package/figures/final/` (override with `CURVE_JSON_DIR` /
`CURVE_FIGURE_DIR`).

---

## 7. Running a single job by hand

Every stage is one `torchrun` invocation. A PAR-shaped PPO run:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

torchrun --standalone --nnodes=1 --nproc-per-node=4 src/train.py \
  --loss_name ppo \
  --model_name gemma2-2b \
  --dataset hh_rlhf \
  --reward_shaping par \
  --exp_name ppo_gemma2-2b_hh_rlhf_par \
  --policy_path    exp_runs/sft_gemma2-2b_hh_rlhf/final_hf \
  --reference_path exp_runs/sft_gemma2-2b_hh_rlhf/final_hf \
  --reward_path    exp_runs/reward_gemma2-2b_hh_rlhf/final_hf \
  --critic_path    exp_runs/reward_gemma2-2b_hh_rlhf/final_hf \
  --n_epochs 0.5 --learning_rate 6e-7 --critic_lr 6e-6 --KL_coef 0.005 \
  --train_batch_size_per_gpu 9 --eval_batch_size_per_gpu 9 \
  --buffer_size 8 --eval_splits_per_epoch 10 \
  --sample_ontest --disable_checkpoint_saving true
```

SFT and reward-model runs need no `*_path` arguments — they start from the base
model in `models_ck/`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 src/train.py \
  --loss_name sft --model_name gemma2-2b --dataset hh_rlhf \
  --exp_name sft_gemma2-2b_hh_rlhf --n_epochs 2 --learning_rate 5e-6

torchrun --standalone --nnodes=1 --nproc-per-node=4 src/train.py \
  --loss_name reward --model_name gemma2-2b --dataset hh_rlhf \
  --exp_name reward_gemma2-2b_hh_rlhf --n_epochs 1 --learning_rate 5e-6 \
  --save_every_eval true --eval_splits_per_epoch 5
```

Arguments worth knowing (full list: `python src/train.py --help`):

| Flag | Meaning |
|---|---|
| `--loss_name` | `sft` / `reward` / `reward_odin` / `ppo` / `a2c` / `grpo` / `dpo` |
| `--reward_shaping` | selects a shaping mode and expands into the underlying `--reward_*` switches via [`reward_shaping_config.py`](src/utils/reward_shaping_config.py) |
| `--num_refs`, `--reward_maxref` | how many reference responses PAR generates / consumes |
| `--reward_ceil` | hard cap on the raw reward before shaping |
| `--KL_coef` | KL-to-reference penalty coefficient |
| `--n_epochs` | fractional values are allowed (`0.1` = 10 % of an epoch) |
| `--eval_splits_per_epoch` | evaluation/sampling points per epoch |
| `--sample_ontest` | write test-set generations for the judge |
| `--eval_only` | evaluate an existing checkpoint without training |
| `--rollout_backend` | `deepspeed` (default) or `vllm_ray` |
| `--disable_checkpoint_saving` | skip all checkpoint writes (used for RL runs) |

`--exp_name` defaults to `<loss>_<model>_<dataset>_<shaping>` and is also the
run directory name. **A run directory is deleted and recreated at startup**
unless `--eval_only` or `--reward_statistics` is set, so rename `--exp_name`
rather than re-running on top of results you want to keep.

---

## 8. Run directory layout and conventions

```text
exp_runs/<exp_name>/
├── train_config.json               # the exact resolved config for the run
├── final_hf/                       # HF export of the final model (SFT/RM only)
├── step_<N>_hf/                    # intermediate exports when --save_every_eval
├── sample_on_test/step_<N>/
│   ├── <rank>.json                 # raw per-rank generations
│   ├── merged.json                 # merged + aligned with dataset prompts
│   └── merged_rated.json           # judge verdicts (after step 4)
├── llm_rating_summary.json
└── llm_rating_curve.png
```

Checkpoint policy, by design, to keep disk usage bounded: RL runs save nothing,
SFT keeps only `final_hf`, and reward runs keep 5 intermediates plus `final_hf`
because WARM needs them.

Batch sizes are resolved per `(phase, objective, model)` by
`_default_batch_size_per_gpu` in [`_common.sh`](src/sbatch/_common.sh); the
tuned H200 defaults live there. Useful overrides:

| Variable | Effect |
|---|---|
| `MAX_BATCH_SIZE_PER_GPU` | global ceiling on per-GPU batch size |
| `AUTO_BATCH_SIZE=0` | disable the OOM-probing autotuner |
| `TRAIN_GPUS`, `ROLLOUT_GPUS` | split GPUs between training and vLLM rollout workers |
| `PAR_TMP_ROOT` | scratch root for TMPDIR / Ray spill / Triton cache |
| `EXP_RUNS_DIR`, `PIPELINE_STATE_ROOT` | relocate run outputs and resume logs |

DeepSpeed ZeRO configs are in `src/configs/stage{2,3}_config.json`; the stage
for each role (policy / reference / reward / critic) is selected in
[`train.py`](src/train.py).

---

## 9. Citation

<!-- TODO: replace with the published reference. -->

```bibtex
@misc{par,
  title  = {Reward Shaping to Mitigate Reward Hacking in RLHF},
  author = {TODO},
  year   = {2025}
}
```

## 10. License

MIT — see [LICENSE](LICENSE).
