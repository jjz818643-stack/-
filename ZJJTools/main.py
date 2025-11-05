import os
import json
import re
import pandas as pd
from typing import Dict
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # 会把 .env 加载到环境变量

# ---------- 配置 ----------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "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")
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}

# ---------- 请求体 ----------
class Patient(BaseModel):
    姓名: str
    年龄: str
    性别: str
    诊断: str
    药品: str
    用量: str

class V1Request(BaseModel):
    patient: Patient

class RefineRequest(BaseModel):
    patient: Patient
    v1: str

# ---------- 辅助 ----------
async def chat(messages, temp=0) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": temp}
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ENDPOINT}/chat/completions",
                            headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ---------- 1. 生成 V1 ----------
async def generate_v1(patient: dict) -> str:
    prompt = f"""
你是儿科临床药师，请按以下模板写一份用药教育，所有小标题内容用“- ”清单呈现。
患者：{json.dumps(patient, ensure_ascii=False)}

模板：
前言（尊敬的XX家长，您好！您的孩子因<原因>，需口服<药名>，该药为<类别>。为了……特此沟通。）

1. 药物作用和目的
2. 剂量和给药时间
3. 不良反应监测
4. 药物相互作用
5. 储存管理
6. 生活方式建议

如有疑问请随时联系儿科医生或药师……祝孩子早日康复！
"""
    return await chat([{"role": "user", "content": prompt}], temp=0.2)

# ---------- 2. 自评 ----------
async def self_refine_education(patient: dict, v1: str) -> str:
    prompt = f"""你是资深儿科临床药师。
请用 3-5 句话、专业且通俗地指出下方 V1 用药教育在准确性、完整性或家长易读性上的具体不足（不要出现模板字面量）。

【患者信息】
{json.dumps(patient, ensure_ascii=False, indent=2)}

【V1 用药教育】
{v1}
输出要求（严格执行）：
- 返回内容只能是**一行纯 JSON**，禁止出现 Markdown 围栏、禁止多行文本；
- 所有长文本里的换行、制表符必须转义成 \\n、\\t；
- 结构如下：
{{"feedback":"真实评价"}}"""
    content = await chat([{"role": "user", "content": prompt}], temp=0)
    content = re.sub(r"```json|```", "", content).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise RuntimeError("LLM 未返回合法 JSON 结构")
    json_str = re.sub(r"[\n\r\t]", " ", m.group(0))
    obj = json.loads(json_str)
    return obj.get("feedback", "")

# ---------- 3. 生成 V3 ----------
async def generate_v3(patient: dict, v1: str, feedback: str) -> str:
    prompt = f"""你是儿科临床药师。
请根据下方「V1 内容」和「审方反馈」重写一份完整、准确、家长友好的 V3 用药教育，必须满足：

1. 禁止出现任何 Markdown 符号（如 ###、***、** 等）；
2. 标题仅用中文数字序号（如“1. 药物作用和目的”），下方用“- ”引出清单；
3. 包含且仅包含以下 6 个小节：
   - 前言（原因、药名、类别、祝愿）
   - 1. 药物作用和目的
   - 2. 剂量和给药时间（含漏服处理、分剂量操作）
   - 3. 不良反应监测（≥3 条常见表现）
   - 4. 药物相互作用（简洁提醒）
   - 5. 储存管理（温度、避光、有效期）
   - 6. 生活方式建议（饮食、作息、复诊）
   - 如有疑问请随时联系儿科医生或药师，我们将持续为您和孩子的健康保驾护航。祝孩子早日康复！
4. 全文不要再出现任何额外解释或总结。

【患者信息】
{json.dumps(patient, ensure_ascii=False, indent=2)}

【V1 内容】
{v1}

【审方反馈】
{feedback}

直接输出最终用药教育全文，不要额外解释。
"""
    return await chat([{"role": "user", "content": prompt}], temp=0.2)

# ---------- FastAPI ----------
app = FastAPI(title="儿科用药教育自迭代 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改成真实域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1")
async def api_v1(req: V1Request):
    try:
        v1 = await generate_v1(req.patient.dict())
        return {"v1": v1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine")
async def api_refine(req: RefineRequest):
    try:
        feedback = await self_refine_education(req.patient.dict(), req.v1)
        v3 = await generate_v3(req.patient.dict(), req.v1, feedback)
        return {"feedback": feedback, "v3": v3}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 健康检查 ----------
@app.get("/ping")
def ping():

    return "pong"
