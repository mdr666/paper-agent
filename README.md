# 学术文献结构化拆解助手

一个基于 **Python + Streamlit + DeepSeek API** 的网页小工具：上传 PDF 论文，自动提取文本，生成结构化拆解，并让 AI 扮演顶会审稿人提出 3 个方法论问题。

## 当前目录结构

第一版刻意保持单文件，先把完整链路跑通：

```text
paper-Agent/
├── .streamlit/
│   └── secrets.toml.example   # API Key 配置示例（复制为 secrets.toml 后填写）
├── .gitignore                 # 排除虚拟环境、真实密钥等
├── app.py                     # 页面 + PDF 解析 + DeepSeek 调用 + Markdown 渲染
├── requirements.txt           # Python 依赖
└── README.md
```

后续功能变多时，建议拆成：

```text
paper-Agent/
├── app.py                     # 只负责页面编排
├── core/
│   ├── pdf_parser.py          # PDF 文本提取、清洗、分块
│   ├── llm.py                 # DeepSeek 客户端、重试与错误处理
│   ├── prompts.py             # 结构化抽取 / 审稿人 Prompt
│   └── schema.py              # 结果字段校验与规范化
├── exports/
│   └── markdown_report.py     # Markdown / 报告导出
└── tests/
    └── test_parser.py
```

## 功能

- 上传文本型 PDF，逐页提取文本并保留页码标记
- 调用 DeepSeek 输出结构化 JSON：
  - 研究背景
  - 核心创新点
  - 方法论
  - 局限性（区分作者承认与 AI 推断）
  - 实验数据（数据集 / 指标 / 结果 / 基线）
- 第二步 Agent：扮演顶会审稿人，提出 3 个具体的方法论漏洞
- 页面以 Markdown 表格展示，并支持下载完整 Markdown 报告
- 处理超长论文：保留正文首 70% + 尾 30%，避免截掉实验与结论

> 单次拆解会发起两次 DeepSeek 请求：一次结构化抽取，一次审稿人质询；论文正文会各自发送一次，请注意 API 成本。

## 快速开始

要求 Python 3.10+。

```powershell
# 在 D:\paper-Agent 下创建虚拟环境
python -m venv .venv

# 安装依赖（不激活虚拟环境也能运行）
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 国内网络如果官方源较慢，可改用清华镜像：
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置 API Key（二选一）
# 项目已经创建了 .streamlit\secrets.toml，直接编辑并填入真实 Key：
notepad .streamlit\secrets.toml

# 如果该文件不存在，再复制示例：
# Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml

# 启动
.\.venv\Scripts\python.exe -m streamlit run app.py
```

也可以不创建 `secrets.toml`，改用环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的Key"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

启动后浏览器会打开 `http://localhost:8501`。如果环境变量和 Secrets 都没有配置，页面左侧会显示密码输入框，可以临时粘贴 Key。

## 配置项

| 名称 | 位置 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | `.streamlit/secrets.toml` 或环境变量 | 必填，DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `.streamlit/secrets.toml` 或环境变量 | 可选，默认 `https://api.deepseek.com` |

默认模型为 `deepseek-chat`，接口地址兼容 OpenAI SDK。

## 工作流

```text
PDF 上传
  → pypdf 逐页提取文本
  → 超长截断（首 70% + 尾 30%）
  → DeepSeek 结构化抽取（JSON）
  → 校验/补齐字段
  → DeepSeek 审稿人质询（论文 + 结构化结果）
  → Markdown 表格 + 审稿问题 + 报告下载
```

## 当前限制

- **扫描版 PDF 不支持**：没有 OCR，提取不到文字时会给出提示。
- **两栏排版可能乱序**：`pypdf` 的文本顺序不总是与阅读顺序一致；后续可换用版面分析或 OCR 方案。
- **超长论文会截断**：当前是首尾截断，不是分块摘要；中段实验细节可能丢失。
- **AI 结果需核查**：模型可能误读数字或把推断当成作者原话，关键结论请回原文确认。
- **隐私**：PDF 正文会发送到 DeepSeek API，请勿上传涉密或未授权文献。

## 安全

`.streamlit/secrets.toml` 和 `.env` 已被 `.gitignore` 排除。不要把真实 API Key 写进 `app.py`、README 或提交记录。
