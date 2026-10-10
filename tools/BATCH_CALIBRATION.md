# 同质 batch=16 校准工具 v1

本工具只补做配对校准，不修改原实验、Repair、权重或测试门槛。
代码和合成测试均在 tools/，不进入原 point-objective-v2 的代码身份列表。
请将整个 tools/ 上传到原服务器项目；保留原有代码、协议、历史清单和运行目录。

## 固定范围

- 读取完整、封存的原 benchmark，以及它绑定的状态和模型；不打开旧 SQLite。
- 原16状态、随机/两规则/开发冻结主模型四策略，共64组。
- 每组固定取 repeat=5..20，即最前16个稳态种子，不根据观察收益挑选。
- 复用原动作、before cost、语义种子、自然输入长度；同一批包含同一状态和动作的16次独立Repair。
- 新增上限1024次Repair，batch16按16次计费，失败/不确定调用保留，不自动重试。
- 成本容差与原检查一致：abs(new-old)<=1e-8*max(1,abs(old))。报告接受收益差异和全部成本不一致对。
- 运行要求原Python版本、NumPy1.26.4、相同PyTorch/CUDA/GPU及确定性配置。
- 不同GPU、历史清单原始字节变化、原代码变化均阻断。请使用原服务器checkout，不用已转换换行的结果副本代替。

## 服务器命令

环境已激活，在 ~/projects/select_01_verify 中执行。check/report只读；run需单独授权1024次Repair和新目录写入。

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -B tools/batch_calibration.py check --source runs/objective-v2-001
```

确认check通过后，授权运行写入 runs/calibration-homogeneous-v1/。
以下使用新目录内的独立日志目录 logs/calibration-homogeneous-v1/，已有日志目录阻断重复启动。

```bash
set +e
bash <<'START'
set -e
cd "$HOME/projects/select_01_verify"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test ! -e runs/calibration-homogeneous-v1
test ! -e logs/calibration-homogeneous-v1
TASK_PY="$(command -v python)"
"$TASK_PY" -B tools/batch_calibration.py check --source runs/objective-v2-001
mkdir -p logs/calibration-homogeneous-v1
nohup bash -c '
"$1" -u -B tools/batch_calibration.py run \
  --source runs/objective-v2-001 --output runs/calibration-homogeneous-v1 \
  --start --cap 1024
rc=$?
echo "EXIT_CODE=$rc"
exit "$rc"
' _ "$TASK_PY" > logs/calibration-homogeneous-v1/run.log 2>&1 < /dev/null &
echo "PID=$!"
START
```

```bash
tail -n 50 logs/calibration-homogeneous-v1/run.log
python -B tools/batch_calibration.py report \
  --source runs/objective-v2-001 --output runs/calibration-homogeneous-v1
```

退出码0：真实校准范围内成本一致。3：采集完整但存在不一致，或只是合成fixture。2：校验/执行失败。
完整但未通过的结果仍封存并可report；失败/中断不自动恢复，硬中断无法封存时按1024披露不确定成本上限。
不得删除旧结果再自动重跑。

## 解释与下一步边界

即使通过，也只支持这64组固定动作、16个种子下的单状态与同质批量一致性。
不证明原异质batch16安全，不证明padding是唯一原因，不重新授权旧freeze-plan，不证明完整NLNS收益。
192实例确认协议及双主比较尚未实现，本次不修改其门槛。

## 合成工程检查

```bash
python -B -m unittest discover -s tools/tests -p 'test_batch_calibration.py' -v
```

该命令会在 tools/tests/_artifacts/ 创建临时合成账本并清理，需写入授权。
测试不导入torch、不加载checkpoint、不执行真实Repair；不能替代服务器校准。

2026-10-10：复用本机 test_02_try 的 Python3.11 环境，9项合成检查全部通过（最终一轮2.563秒）。
覆盖原动作/自然长度重建、固定稳态种子、不一致和错误padding阻断、完整1024调用账本、
失败批次计费、不覆盖、不越界、只读报告，以及tools不进入旧代码身份。
尚未读取本次真实服务器结果，也未在服务器加载checkpoint或执行真实校准。
