# AutoFinetune 使用指南

## 目前進度（2026-03-22）

### 已完成
| 項目 | 狀態 |
|------|------|
| AutoFinetune 程式碼 commit 到 repo | ✅ |
| README 更新（加入 AutoFinetune 說明） | ✅ |
| mamba 環境 `autofinetune` 建立完成 | ✅ |
| `finetune.py` API 相容性修正（trl v0.29+） | ✅ |
| 實驗 branch `autofinetune/mar22` 建立 | ✅ |

### 尚未完成
- Baseline 實驗尚未成功執行（第一次嘗試因 API bug crash，已修復，等待重新執行）

---

## 實驗會如何運行

### 整體架構
```
agent（你/Claude）
    ↓ 修改 finetune.py
    ↓ git commit
orchestrate.py
    ↓ 執行 finetune.py（QLoRA 訓練 Qwen3-8B，~20 分鐘）
    ↓ 執行 eval.py（IFEval + MATH-500 + HumanEval+，~16 分鐘）
    ↓ composite_score > 歷史最佳？
        YES → 保留 commit（keep）
        NO  → git reset --hard HEAD~1（discard）
    ↓ 寫入 results.tsv
```

### 評分指標
- **composite_score** = (IFEval + MATH-500 + HumanEval+) / 3，範圍 [0, 1]，越高越好
- IFEval：指令跟隨（格式約束）
- MATH-500：數學推理（500 題競賽數學）
- HumanEval+：程式生成（164 題 Python）

### 每次實驗時間
- ~36 分鐘（20 分鐘訓練 + 16 分鐘評估）
- 超過 60 分鐘視為 timeout

---

## 手動操作步驟

### 1. 連線並進入 tmux session

```bash
# SSH 連進機器後，新建 tmux session（第一次）
tmux new -s autofinetune

# 之後重新連線，attach 回已有 session
tmux attach -t autofinetune

# 列出所有 session
tmux ls

# tmux 內部分離（不關掉）
Ctrl+B  d
```

### 2. 切換到正確環境與目錄

```bash
cd ~/claude_space/autoresearch
conda activate autofinetune
# 或
mamba activate autofinetune
```

> **重建環境（第一次或環境損毀時）：**
> ```bash
> mamba env create -f environment.yml
> mamba activate autofinetune
> pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
> ```

### 3. 執行 Baseline（第一次啟動）

```bash
# 在 tmux 內執行，這會先跑 baseline 再等 agent 接手
python orchestrate.py agent --fast 2>&1 | tee run.log
```

Baseline 完成後會印出：
```
[orchestrate] Agent loop ready.
[orchestrate] Current best composite: <數字>
```

### 4. 看目前結果

```bash
# 查看所有實驗結果
cat results.tsv

# 查看最佳分數
python orchestrate.py status

# 即時追蹤 log
tail -f run.log
```

### 5. 手動執行單一實驗（agent 模式）

```bash
# 步驟：先修改 finetune.py，然後執行
python orchestrate.py run-one <實驗編號> "<說明>" --fast

# 範例
python orchestrate.py run-one 1 "add MetaMathQA for math improvement" --fast
python orchestrate.py run-one 2 "increase lora rank to 32" --fast
```

### 6. 查看實驗 log（debug 用）

```bash
# 訓練階段 log
cat run.log | tail -50

# 評估階段 log
cat eval.log | tail -50

# 只看分數
grep "composite_score\|ifeval_strict\|math_500_em\|humaneval_pass1" eval.log
```

### 7. 查看 git 實驗歷史

```bash
# 看實驗 commits
git log --oneline

# 看目前在哪個 branch
git branch

# 看 finetune.py 目前設定
cat finetune.py | head -80
```

---

## 檔案說明

```
finetune.py          ← Agent 唯一修改的檔案（LoRA 設定、資料集、超參數）
eval.py              ← 評估 oracle（禁止修改）
orchestrate.py       ← 實驗流程基礎設施（禁止修改）
optuna_runner.py     ← Optuna 數值 HPO 基線（可參考）
program_finetune.md  ← Agent 策略指引
results.tsv          ← 實驗結果紀錄（不 commit 到 git）
run.log              ← 最新一次訓練 log
eval.log             ← 最新一次評估 log
output/              ← 訓練輸出（adapter 權重等）
```

---

## Branch 說明

```
master              ← 主要程式碼（AutoResearch + AutoFinetune 基礎框架）
autofinetune/mar22  ← 實驗 branch（所有實驗 commits 都在這裡）
```

---

## 常見問題

### 實驗 crash 怎麼辦？
orchestrate.py 會自動記錄 `crash` 並 revert commit，直接執行下一個實驗即可。

### 想重置到最佳狀態？
```bash
# 找到最後一個 keep 的 commit hash（從 results.tsv）
git log --oneline | grep -f <(awk '$7=="keep"{print $1}' results.tsv)
```

### mamba 環境的 Python 位置
```bash
which python  # 確認是否在正確環境
# 應該是 ~/miniforge3/envs/autofinetune/bin/python
```

### 磁碟空間不足？
output/ 目錄會堆積 adapter 檔案，可清理舊的：
```bash
ls -lh output/
rm -rf output/run_000{0..5}/  # 視情況刪除舊的
```
