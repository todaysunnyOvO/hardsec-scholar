# HardSec Scholar 本地运行指南

HardSec Scholar 默认只监听本机地址。论文副本、SQLite 数据库、Chroma 索引、日志和模型密钥都保存在本地，不会由项目自动发布到托管环境。

## 1. 环境与安装

需要 Python 3.10+、Node.js 22+ 和 npm。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

cd web
npm ci
cd ..
```

## 2. 配置模型

```powershell
Copy-Item .env.example .env
```

使用已验证的 DeepSeek 生成模型和 SiliconFlow 向量服务时填写：

```env
LLM_PROVIDER=openai
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY=your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=your-siliconflow-key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1

ALLOW_WEB_SEARCH=false
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`LLM_PROVIDER=openai` 和 `EMBEDDING_PROVIDER=openai` 表示项目调用 OpenAI 兼容协议。两类模型可由不同供应商提供，也使用不同密钥。DeepSeek 官方地址会自动关闭 Thinking，以兼容强制 function calling 的结构化输出。

不要在终端截图、日志、提交记录或 README 中粘贴真实密钥。`.env` 已加入 `.gitignore`。

## 3. 一键启停

```powershell
.\scripts\start_local.ps1
```

脚本会先检查 `.venv`、`.env`、前端依赖和端口占用，再以隐藏窗口启动两个本地服务，并等待健康检查通过：

- 网页：`http://localhost:3000`
- API：`http://127.0.0.1:8000`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`
- 日志：`logs/`

停止由脚本启动的进程：

```powershell
.\scripts\stop_local.ps1
```

停止脚本只处理状态文件中记录、且启动时间一致的进程树，避免误杀复用同一 PID 的其他程序。状态文件位于 `data/run/`，不会提交到 Git。

如果 PowerShell 阻止脚本，只为当前窗口临时允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 4. 手动启动

需要调试日志或前台断点时，打开两个终端。

终端一：

```powershell
.\.venv\Scripts\python.exe -m uvicorn hardsec_scholar.api.app:app --host 127.0.0.1 --port 8000 --reload
```

终端二：

```powershell
cd web
npm run dev
```

## 5. 使用顺序

1. 在 Paper library 上传英文文本型 PDF。
2. 确认状态为 `indexed`。若为 `parsed`，检查 Embedding 配置后点击 Reindex。
3. 在 Research 选择论文；不选择时检索全部已索引论文。
4. 输入问题，查看逐步出现的 Agent 轨迹。
5. 点击 Evidence ID，核对论文标题、章节、页码和原文片段。
6. 用语料中没有答案的问题验证拒答边界。
7. 默认开启 **Save conversation history**；关闭后本次问答只保留在当前页面，不写入 SQLite。
8. 在 **History** 页面查看完整消息、继续已有会话，或删除会话及其运行和轨迹。

扫描 PDF 暂无 OCR。Web 搜索默认关闭，因此回答只基于本地论文。

## 6. 常见问题

| 现象 | 检查方式 |
| --- | --- |
| `Port 3000/8000 is already in use` | 先运行停止脚本；若仍占用，用 `Get-NetTCPConnection -State Listen -LocalPort 3000,8000` 查看进程 |
| 上传后为 `parsed` | 检查 Embedding 模型、密钥和 Base URL，再执行 Reindex |
| 问答返回 503 | 检查 LLM/Embedding 配置是否为空，供应商余额和网络是否可用 |
| API 正常但网页连接失败 | 检查 `NEXT_PUBLIC_API_URL`，默认应为 `http://localhost:8000` |
| 一键启动超时 | 查看 `logs/api.stderr.log` 和 `logs/web.stderr.log`；修复后先运行停止脚本 |
| 扫描论文无正文 | 当前版本不含 OCR，需换用文本型 PDF 或先在外部完成 OCR |

## 7. 质量检查

```powershell
.\.venv\Scripts\python.exe -m ruff check src/hardsec_scholar tests/conftest.py tests/unit tests/integration scripts
.\.venv\Scripts\python.exe -m mypy src/hardsec_scholar scripts
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q

cd web
npm run lint
npm test
```
