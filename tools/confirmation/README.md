# 冻结分类器确认实验：服务器使用说明

这是一套独立的新确认流程。旧 objective-v2-001 和 calibration-homogeneous-v1 只读。
协议见 protocol.md；策略文件 policy192.json 固定192实例、新seed、两个主要比较。
没有训练入口，也没有通过修改旧门槛绕过旧freeze-plan。

## 同步与检查

将 tools/ 的本次新增文件提交并同步到原服务器checkout；不要修改原根目录、point_select/、tests/、protocol/。
旧校准工具 batch_calibration.py 没有修改，因此已封存校准的工具哈希仍有效。
运行中的确认代码身份包括新协议、策略、Python文件、原校准工具和启动脚本；开始后不可改动。

```bash
cd ~/projects/select_01_verify
source "$HOME/projects/test_02_try/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -B tools/confirm_point.py budget
python -B tools/confirm_point.py check
```

check只读：重新核对旧来源和校准、冻结模型规格、开发评价与batch1成本、双比较样本量。
只有 may_freeze_plan=true 才能冻结确认计划。若建议N>192，停止评审，不自动扩大N或预算。
check不加载checkpoint、不运行真实Repair，也不生成新实例。生产真实阶段另检查原Python/NumPy、
torch/CUDA/GPU/确定性设置与原benchmark一致；不会静默CPU回退。
当前历史清单只有已知12旧实例，仍不宣称全历史排重。

## 授权后的后台运行

以下仅为手动命令模板，真实计算预算尚需明确授权。
写入新 runs/classifier-confirm-v1-192/ 和 logs/classifier-confirm-v1-192/。
上限1021440次Repair（状态38400、A196608、B786432），去重可降低。旧开发和校准开销不抵扣。
看到PID后可断开SSH，nohup处理挂断；脚本错误只停止子进程，不退出登录shell。
服务器若配置了注销时强制清理用户进程，应使用管理员提供的持久作业系统。

```bash
set +e
bash <<'START'
set -e
cd "$HOME/projects/select_01_verify"
test ! -e runs/classifier-confirm-v1-192
test ! -e logs/classifier-confirm-v1-192
export CONFIRM_PYTHON="$HOME/projects/test_02_try/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$CONFIRM_PYTHON" -B tools/confirm_point.py check
mkdir -p logs/classifier-confirm-v1-192
nohup bash -c '
bash tools/run_confirmation192.sh --start --cap-total 1021440
rc=$?
echo "EXIT_CODE=$rc"
exit "$rc"
' > logs/classifier-confirm-v1-192/run.log 2>&1 < /dev/null &
echo "已后台启动，PID=$!"
START
```

脚本依次冻结计划、生成状态/候选/特征、冻结预测、采A32、冻结A参照、采B128、审计和导出报告。
每阶段失败即停止；不启动完整NLNS。
`CONFIRMATION_COMPLETE: EXIT_CODE=0` 表示工程流程完整完成，不表示统计门槛通过。
成功/失败的研究结论读report中的interpretation，不看脚本退出码代替研究判断。

```bash
tail -n 50 ~/projects/select_01_verify/logs/classifier-confirm-v1-192/run.log
pgrep -af '[c]onfirm_point.py' || true
nvidia-smi
```

每16个实例或状态输出进度，沉默期间不要重复启动。
失败/中断保留所有目录：先只读audit，不删除、不自动重试、不改变seed。
audit遇到未封存账本只报告该阶段cap的不确定上限，不打开SQLite。

```bash
"$HOME/projects/test_02_try/.venv/bin/python" -B tools/confirm_point.py audit
"$HOME/projects/test_02_try/.venv/bin/python" -B tools/confirm_point.py analyze
```

完整报告由后台脚本排他导出到 runs/classifier-confirm-v1-192/report.json。
analyze默认精简输出，--full输出完整JSON；只读analyze不会落盘，导出必须--start且输出是新run根目录新文件。
既有report不能覆盖。

## 单阶段入口

不使用整段脚本时，按上述阶段顺序逐条授权运行：

```bash
python -B tools/confirm_point.py freeze-plan --start
python -B tools/confirm_point.py test-states --start --cap 38400
python -B tools/confirm_point.py test-candidates --start
python -B tools/confirm_point.py test-features --start
python -B tools/confirm_point.py freeze-test-predictions --start
python -B tools/confirm_point.py collect-test-a --start --cap 196608
python -B tools/confirm_point.py freeze-test-reference --start
python -B tools/confirm_point.py collect-test-b --start --cap 786432
```

不要在后台脚本仍运行时执行这些阶段。只支持新run或从完整封存阶段继续下一个未开始阶段，
不是失败阶段恢复工具。默认source/calibration/run可用对应参数显式指定，身份必须匹配。

## 证据解释

主比较单侧配对t alpha=.025，各自95%实例bootstrap敏感性，实用均值门槛.01pp。
192规划备择为.05pp，不是以.01为零假设证明真实优势>.01，亦不保证两比较联合80%功效。
时间代理只用旧开发batch1完整路径；新测试没有另做延迟测量。
通过只建议另行评审等墙钟完整搜索，不能以次要模型/第50步替换主目标。
未通过时报告区间和成本分别判断，不自动加标签、扩模型、扩样或执行搜索。

合成工程检查命令（会在 tools/tests/_artifacts/ 写入并清理临时fixture，需写入授权）：

```bash
python -B -m unittest discover -s tools/tests -p 'test_*.py' -v
```

工程fixture仅测试机械流程，不导入torch或执行真实Repair，不产生确认科学证据。

## 本地验证记录

2026-10-10：复用 test_02_try 的Python3.11环境，tools/tests下18项校准及确认合成测试
同一轮全部通过，21.537秒。确认流程使用4个合成实例缩小工程范围，并仅替换外部来源与模拟后端；
真实CLI没有fixture绕过参数。额外只读budget入口返回192实例、0次拟合、1021440次Repair上限。
未修改任何旧追踪文件；未加载真实checkpoint、未在服务器运行确认采集，Bash启动脚本尚未在服务器执行。
