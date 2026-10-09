# 固定 Repair：Point 评分目标对照

Windows/Linux独立实验包。固定E4原生Point、K20、12项特征和官方P20% Repair，对照接受收益Ridge、状态内Ridge、线性排序、受限GBDT和LENS式改善事件分类。协议见 [POINT_OBJECTIVE_PROTOCOL.md](protocol/POINT_OBJECTIVE_PROTOCOL.md)。没有Repair训练或完整搜索入口。

本仓库只有代码和模板，**没有真实实验结果或批准预算**。旧实验保留在原项目。权重已复制并检查SHA256，来源和GPL许可一并保留。

## 只读检查

Python>=3.11；真实阶段要求NumPy==1.26.4以及预先配置的PyTorch>=2.5，V100 wheel须支持sm_70。模型仅用NumPy，不需要sklearn/scipy。不会自动安装依赖或CPU回退。

```powershell
python -B point_select_cli.py check
python -B point_select_cli.py budget
python -B point_select_cli.py budget --test-instances 96
```

check只读AST、安装元数据和权重hash，不导入NumPy/torch、不加载权重、不开GPU、不生成缓存或目录。默认模板64开发/16面板，不默认测试数量。64/96/320仅预算情景，不是批准的实例数。

## 写入与逐阶段预算

先取得代码/配置/运行目录写入授权，再复制 `configs/pilot.template.json` 为实际配置，例如 `configs/pilot.json`；明确设备、root seed、每阶段cap。不要修改已开始run绑定的配置、源码、协议、历史或权重。输出只能在项目内，禁止指向外部的符号链接。

以下是**待逐条授权并手动启动的命令模板**，不要编成一键脚本。默认run示例 `runs/objective-v2-001`。

| 阶段 | 显式参数 | 默认去重前Repair上限 |
|---|---|---:|
| dev-states | --start --cap 12800 | 12800 |
| dev-candidates / dev-features | --start | 0 |
| collect-dev-a | --start --cap 65536 | 65536 |
| fit-oof | --start --max-fits 60 | 0，60 fits |
| collect-dev-v | --start --cap 131072 | 131072 |
| collect-panel-a | --start --cap 16384 | 16384 |
| collect-panel-v | --start --cap 32768 | 32768 |
| select-models | --start --max-fits 21 | 0，5最终+16覆盖fits |
| diagnostics | --start | 0 |
| benchmark | --start --cap 3200 | 3200，含预热 |
| freeze-plan | --start --policy 实际policy | 0 |
| test-states | --start --cap 200N | 200N |
| test-candidates / test-features | --start | 0 |
| freeze-test-predictions | --start | 0 |
| collect-test-a | --start --cap 1024N | 1024N |
| freeze-test-reference | --start | 0 |
| collect-test-b | --start --cap 4096N | 4096N |

例：

```powershell
python -B point_select_cli.py dev-states --config configs/pilot.json --run runs/objective-v2-001 --start --cap 12800
```

其他阶段替换command及对应参数。每条命令均独立审批，不自动消费全预算。batch16仍收费16次Repair。实际候选去重可能降低cap所需下限；失败/不确定调用不免费，不自动重试。

## 新测试之前的门槛

diagnostics和benchmark完成后，先只读查看：

```powershell
python -B point_select_cli.py assess --config configs/pilot.json --run runs/objective-v2-001
```

它返回独立V选择收益、实例方差、建议测试N、真实路径成本和继续门槛。默认policy模板test_instances=null，不会启动测试。若开发M未超过随机及开发最强规则，或E[z]/完整时间代理未通过1.05，freeze-plan阻断。

在看新测试前，经预算评审复制 `configs/decision.policy.template.json` 为实际policy，填入>=建议数量且为16倍数的test_instances，冻结部署batch1/16。阈值不允许临时修改。freeze-plan前不生成新测试状态；数据不足也不能通过“反复加Repair”绕过N要求。

测试预测及完整分数先冻结，随后测试A32、A参照、B128。期望随机平局为主，一次冻结随机选择保留；常数期望必须等于随机期望。测试后禁止改主模型、阶段或随机种子。

## 只读审计与分析

```powershell
python -B point_select_cli.py audit --config configs/pilot.json --run runs/objective-v2-001
python -B point_select_cli.py analyze --config configs/pilot.json --run runs/objective-v2-001
```

audit读sealed receipts/JSONL exports，不打开SQLite/WAL。检查阶段顺序、源码/权重、实例分区、候选、特征、标签种子、逐次z、冻结分数和平局、A参照及全部成本。analyze默认输出紧凑主结论；--full输出完整报告。

授权导出后：

```powershell
python -B point_select_cli.py analyze --config configs/pilot.json --run runs/objective-v2-001 --start --output runs/objective-v2-001/report.json
```

三个主比较：WR−AR、冻结M−随机、M−冻结S*。单侧配对t Bonferroni alpha=.05/3，95%实例bootstrap敏感性；项目收益均值>=.01pp、下界>0和成本代理通过，才建议单独评审等时间搜索。不能据此证明真实收益>.01，也不能以次要测试赢家替换M。

## 工程验证与中断

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

这是写入测试：合成临时文件仅在 `tests/_artifacts/`，完成后清理；合成模型fit不是研究训练，不导入torch/真实Repair。验证记录见 `VALIDATION.md`。

started/failed/不完整阶段不可覆盖或自动重启。保留原目录，先只读audit；未封存SQLite须另行恢复审计和授权，未知调用按cap披露，不能当零开销。更改研究方案使用新协议、seed和目录；历史已查看数据不能重新成为确认集。
