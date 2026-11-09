# DeepLISA

[简体中文](README_CN.md) ｜ [English](README_EN.md)

DeepLISA 是一个面向荧光微腔室图像的 AI 辅助、物理约束型图像分析流程，用于将数字微腔室实验中的阳性荧光腔室转换为可审计的计数结果、Poisson 占有率和基于校准曲线的浓度估计。该程序最初用于 dPAC-DeepLISA 体系中的微腔室读数分析，也可作为类似数字免疫分析或数字 CRISPR 读数实验的图像分析参考框架。

DeepLISA 不是临床诊断软件，也不声称可直接用于临床样本判读。公开代码主要用于科研复现、方法审计和图像分析流程展示。


## 30 秒判断能不能用

- **给谁**：做数字免疫分析 / 数字 CRISPR 读数、需要逐个核对阳性腔而不是接受一个黑箱数字的人。
- **解决什么**：把荧光微腔室图像变成可审计的计数——阳性腔数 K、Poisson 占有率 λ、浓度估计，并附 overlay 图、候选表与批量 QC 报告。
- **怎么验证**：打开 overlay 图，每个被判阳性的腔都应能在图上找到；对照候选表核几个点；看 QC 报告里背景/饱和/焦面是否在可接受范围。若 overlay 与实际图像不符，本次结果视为无效。

## 核心功能

- 荧光微腔室图像导入和批量处理。
- Top-hat 背景校正、CLAHE 局部对比度增强、Gaussian 平滑和自适应阈值分割。
- 亮点优先的阳性候选检测，包括连通域提取、距离变换分离和轮廓候选生成。
- 基于局部信号背景比、局部 z-score、support fraction、面积、圆度和半径一致性的候选评分。
- 可选网格一致性检查，用于辅助判断腔室阵列图像中的候选点是否合理。
- PSF-inspired 局部信号一致性检查和图像级质量控制。
- 输出阳性腔室数 K、Poisson 占有率 lambda、估算浓度、结果 overlay 图、候选表和批量 QC 报告。
- 支持单图分析、文件夹批处理和候选点审计导出。

## 适用范围

本程序适合分析经过荧光显微成像获得的数字微腔室图像，尤其是背景较低、腔室尺寸相对稳定、阳性信号呈局部圆形或近圆形亮点的图像。

对于局部截图、严重失焦、气泡污染、强饱和亮斑、明显黑边、非均匀照明或芯片区域不完整的图像，DeepLISA 仍可能给出候选计数，但这类结果应结合 overlay 图和 QC 指标人工复核，不建议直接作为正式定量结果。

## 安装环境

推荐环境：

- Python 3.9 或更高版本
- Windows 10/11、macOS 或 Linux

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `opencv-python`
- `numpy`
- `scipy`
- `Pillow`
- `tkinterdnd2`
- `python-dotenv`
- `requests`

其中 `requests` 和 `python-dotenv` 主要用于可选外部参数建议分支。默认图像分析流程不依赖私有 API。

## 快速开始

### 启动图形界面

在代码目录中运行：

```bash
python main.py
```

图形界面支持：

- 导入单张图像
- 批量处理文件夹
- 设置腔室半径、腔室总数、腔室体积和分子量等参数
- 开启或关闭局部 ROI 模式
- 查看结果 overlay 图和分析日志

### 命令行导出候选点表

导出单张图像或文件夹中的候选点记录：

```bash
python export_candidate_table.py <image-or-folder> --out results/candidate_export
```

使用本地规则分析，不调用可选外部参数建议：

```bash
python export_candidate_table.py <image-or-folder> --out results/candidate_export --no-ai
```

分析局部 ROI 截图：

```bash
python export_candidate_table.py <image-or-folder> --out results/local_roi_export --local-roi
```

常用参数：

```bash
--radius 7
--mw 4.3
--volume-pl 785
--n-total 21600
```

其中 `n-total` 默认为单个检测区域的理论腔室数。局部 ROI 图像通常只适合用于候选点和局部计数检查，不建议直接用于正式浓度推断。

## 输出文件

批处理或候选导出通常会生成以下文件：

- `result_<filename>.jpg`：带阳性候选标记的 overlay 结果图。
- `batch_report.csv`：每张图的阳性腔室数、估算浓度、参数、置信度和 QC 指标。
- `candidate_table.csv`：被接受的阳性候选点记录。
- `all_candidate_table.csv`：包括 accepted、weak、uncertain 和 artifact 等类别的候选记录。
- `*_convergence.json`：参数迭代、计数和质量指标记录。
- `spatial_quadrat_counts.csv` 和 `spatial_quadrat_residuals.csv`：空间 QC 统计结果，若启用相关导出。

这些输出文件用于审计图像分析过程。正式定量时，应同时检查计数结果、overlay 图和 QC 字段。

## 主要文件说明

```text
main.py                         GUI 入口
sample_analysis.py              单图和批量分析调度
processing_pipeline.py           核心图像处理和候选检测流程
advanced_filtering.py            图像增强和基础筛选工具
image_utils.py                   图像读取、保存和 overlay 绘制
concentration_calculator.py      Poisson 占有率和浓度换算
quality_control.py               基础质量控制工具
psf_verification.py              PSF-inspired 信号一致性检查
grid_detection.py                网格辅助检查和阵列几何处理
region_inference.py              单区域/多区域图像范围推断
spatial_qc.py                    空间分布和 chamber-opportunity 加权 QC
design_prior.py                  芯片设计先验读取和整理
agent_core.py                    参数迭代和分析流程控制
agent_quality.py                 综合置信度和质量评分
agent_tools.py                   分析工具封装
agent_batch_context.py           批量一致性上下文
export_candidate_table.py        候选点和审计表导出命令行工具
config.py                        默认参数和环境变量配置
```

## 环境变量

DeepLISA 可通过环境变量覆盖部分默认参数。常用变量包括：

```text
RESULT_DIR
TEMP_DIR
RADIUS
DEFAULT_MW_KDA
DEFAULT_VOL_PL
DEFAULT_TOTAL_CHAMBERS
LANGUAGE
AUTO_REGION_INFERENCE
DETECTION_MODE
EXPORT_DETECTION_AUDIT
EXPORT_GRID_CANDIDATES
```

可选外部参数建议分支可读取：

```text
GLM_API_KEY
GLM_BASE_URL
AI_MODEL
```

公开仓库中不要上传 `.env` 文件或任何私有 API key。若需要提供示例配置，请使用 `.env.example`，并确保里面没有真实密钥。

## 公开上传建议

建议上传源代码和必要说明文件，不要上传以下内容：

```text
.env
.venv/
.idea/
__pycache__/
results/
tmp/
原始显微图像
批处理输出文件
内部修复记录
包含本地路径或私有信息的日志
```

如果要发布 GitHub 版本，建议额外加入：

```text
README_CN.md
README_EN.md
requirements.txt
.gitignore
LICENSE
examples/README.md
```

## 重要说明

- DeepLISA 输出的是图像分析结果，不应脱离实验设计、校准曲线和 QC 指标独立解释。
- 自动计数结果应通过 overlay 图进行抽查，尤其是在低信噪比、高背景或局部 ROI 图像中。
- 可选外部参数建议分支不是复现论文图像分析结果的必要条件。
- 若用于论文、报告或公开数据，请同时提供版本号、参数设置、候选表和批量报告，便于他人复核。

## 引用

如果使用 DeepLISA 或其中的图像分析流程，请引用相关 dPAC-DeepLISA 论文或代码仓库。正式 DOI 或论文信息发布后，可在此处补充。

## 许可

请在公开发布前补充明确的 `LICENSE` 文件。若未提供许可证，则默认不授予第三方复制、修改或再分发权限。


## 许可

本项目采用 MIT 许可证，见 [LICENSE](LICENSE)。

## 成果状态与引用

本代码是 dPAC–DeepLISA 数字微腔室检测方法学的配套分析流程，配套方法学论文处于审稿阶段。
若用于发表，请引用本仓库并注明所用版本（tag）。

## 复核方式

1. 按 `requirements.txt` 安装依赖，运行 `python main.py` 对单图或文件夹批处理。
2. 打开输出目录中的 overlay 图与候选表：每个阳性腔应可在图上逐个核对。
3. 查看批量 QC 报告，确认图像级指标（背景、饱和度、焦面）在可接受范围。
