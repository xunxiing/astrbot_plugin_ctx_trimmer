# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import AstrBotConfig, logger

# 可选依赖：tiktoken（OpenAI/兼容系精确计数）
try:
    import tiktoken  # type: ignore
except Exception:  # 环境里没有就走估算
    tiktoken = None


def _flatten_text(content: Union[str, List[Dict[str, Any]], Dict[str, Any], None]) -> str:
    """将 OpenAI 风格的 content 统一拍平成字符串，只取对 token 有意义的可见文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # 兼容 gemini/claude 的一些块结构
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        buf: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("type")
                if t == "text" and isinstance(part.get("text"), str):
                    buf.append(part["text"])
                elif t in ("input_text", "output_text") and isinstance(part.get("content"), str):
                    buf.append(part["content"])
                # image/audio等忽略，不计文本token
            else:
                buf.append(str(part))
        return "\n".join(buf)
    return str(content)


def _guess_openai_encoding(model: Optional[str]) -> "tiktoken.Encoding":
    """根据模型名猜测 tiktoken 编码；失败则退回 cl100k_base。"""
    assert tiktoken is not None
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens_openai_style(messages: List[Dict[str, Any]], system_prompt: str, model: Optional[str]) -> int:
    """
    近似计数：OpenAI/兼容聊天格式，每条消息有头部开销。
    不追求绝对精确，但足够用于裁剪。
    """
    # 参考 OpenAI chat 计算思路；不同模型有常数开销的细微差异，这里统一近似。
    if tiktoken is None:
        # 粗略估算：4 字符 ≈ 1 token
        total_text = system_prompt
        for m in messages:
            total_text += _flatten_text(m.get("content", ""))
            if "tool_calls" in m:
                total_text += json.dumps(m["tool_calls"], ensure_ascii=False)
            if "name" in m:
                total_text += str(m["name"])
        return max(1, len(total_text) // 4)

    enc = _guess_openai_encoding(model)
    tokens = 0
    # system
    tokens += len(enc.encode(system_prompt or ""))

    # per-message 开销：按 OpenAI 经验给一个常量近似
    PER_MSG_OVERHEAD = 4
    PER_NAME_OVERHEAD = -1  # 有 name 时替换 role 开销
    for m in messages:
        tokens += PER_MSG_OVERHEAD
        content_text = _flatten_text(m.get("content"))
        tokens += len(enc.encode(content_text))
        if m.get("name"):
            tokens += PER_NAME_OVERHEAD
        # 简单把 tool_calls 序列化后计入
        if m.get("tool_calls"):
            tokens += len(enc.encode(json.dumps(m["tool_calls"], ensure_ascii=False)))
        if m.get("function_call"):
            tokens += len(enc.encode(json.dumps(m["function_call"], ensure_ascii=False)))
    return tokens


def _drop_old_turns(contexts: List[Dict[str, Any]], turns: int) -> List[Dict[str, Any]]:
    """
    以“用户发言”为轮次边界，丢弃最早的 N 轮对话：
    从最早的 user 开始，删除直到第 N+1 个 user 出现的前一个位置（含期间的 assistant/tool）。
    """
    if turns <= 0 or not contexts:
        return contexts

    user_indices = [i for i, m in enumerate(contexts) if (m.get("role") == "user")]
    if not user_indices:
        # 没有 user，直接清空更安全
        return []

    k = min(turns, len(user_indices))
    # 如果 k 等于现有 user 轮数，说明要丢到末尾
    if k == len(user_indices):
        return []

    cut_to = user_indices[k]  # 第 k+1 个 user 的索引位置
    return contexts[cut_to:]


def _build_messages_for_count(req: ProviderRequest) -> List[Dict[str, Any]]:
    """
    计数时构造最终 messages 视图：system 独立，contexts + 当前 prompt。
    注意：不修改 req，仅用于估算。
    """
    messages = list(req.contexts or [])
    if getattr(req, "prompt", None):
        messages = messages + [{"role": "user", "content": req.prompt}]
    return messages


@register("ctx_trimmer", "you", "基于 token 的上下文裁剪器：保留 system，按对话轮丢弃历史", "0.1.0", "")
class CtxTrimmer(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config or AstrBotConfig()

        # 中英文键名映射（兼容旧版本）
        KEYMAP = [
            ("启用", "enabled"),
            ("最大上下文token", "max_context_tokens"),
            ("每次丢弃的对话轮数", "drop_turns_each_time"),
            ("输出保留余量token", "output_margin_tokens"),
            ("计数模型提示", "count_model_hint"),
        ]

        # 自动迁移：如果中文键不存在但英文键存在，则搬迁；最后删除英文键，统一存中文
        migrated = False
        for cn, en in KEYMAP:
            if (cn not in self.config) and (en in self.config):
                self.config[cn] = self.config[en]
                migrated = True
        # 清理旧英文键，避免以后读到两份
        for _, en in KEYMAP:
            if en in self.config:
                try:
                    del self.config[en]
                    migrated = True
                except Exception:
                    pass
        if migrated:
            self.config.save_config()

        def _g(cn, default):
            return self.config.get(cn, default)

        self.enabled: bool = bool(_g("启用", True))
        self.max_context_tokens: int = int(_g("最大上下文token", 8192))
        self.drop_turns_each_time: int = int(_g("每次丢弃的对话轮数", 2))
        self.output_margin_tokens: int = int(_g("输出保留余量token", 512))
        self.count_model_hint: Optional[str] = _g("计数模型提示", "") or None

    # === 指令组：/ctxlimit ===
    @filter.command_group("ctxlimit")
    def ctxlimit(self):
        """上下文裁剪器配置"""

    @filter.command_group("上下文")
    def ctxlimit_cn(self):
        """上下文裁剪器配置（中文别名）"""

    @ctxlimit_cn.command("状态")
    async def cmd_status_cn(self, event: AstrMessageEvent):
        async for x in self.cmd_status(event):  # 直接复用
            yield x

    @ctxlimit_cn.command("启用")
    async def cmd_enable_cn(self, event: AstrMessageEvent):
        async for x in self.cmd_enable(event):
            yield x

    @ctxlimit_cn.command("停用")
    async def cmd_disable_cn(self, event: AstrMessageEvent):
        async for x in self.cmd_disable(event):
            yield x

    @ctxlimit_cn.command("设最大")
    async def cmd_setmax_cn(self, event: AstrMessageEvent, 最大: int):
        async for x in self.cmd_setmax(event, 最大):
            yield x

    @ctxlimit_cn.command("设丢弃")
    async def cmd_setdrop_cn(self, event: AstrMessageEvent, 轮数: int):
        async for x in self.cmd_setdrop(event, 轮数):
            yield x

    @ctxlimit_cn.command("设余量")
    async def cmd_setmargin_cn(self, event: AstrMessageEvent, 余量: int):
        async for x in self.cmd_setmargin(event, 余量):
            yield x

    @ctxlimit.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        msg = (
            f"启用: {self.enabled}\n"
            f"最大上下文 token: {self.max_context_tokens}\n"
            f"每次丢弃的对话轮数: {self.drop_turns_each_time}\n"
            f"输出保留余量 token: {self.output_margin_tokens}\n"
            f"计数模型提示: {self.count_model_hint or '(自动)'}"
        )
        yield event.plain_result(msg)

    @ctxlimit.command("enable")
    async def cmd_enable(self, event: AstrMessageEvent):
        self.enabled = True
        self.config["启用"] = True
        self.config.save_config()
        yield event.plain_result("已启用。")

    @ctxlimit.command("disable")
    async def cmd_disable(self, event: AstrMessageEvent):
        self.enabled = False
        self.config["启用"] = False
        self.config.save_config()
        yield event.plain_result("已停用。")

    @ctxlimit.command("setmax")
    async def cmd_setmax(self, event: AstrMessageEvent, max_tokens: int):
        self.max_context_tokens = max(512, int(max_tokens))
        self.config["最大上下文token"] = self.max_context_tokens
        self.config.save_config()
        yield event.plain_result(f"最大上下文 token 已设为 {self.max_context_tokens}")

    @ctxlimit.command("setdrop")
    async def cmd_setdrop(self, event: AstrMessageEvent, turns: int):
        self.drop_turns_each_time = max(1, int(turns))
        self.config["每次丢弃的对话轮数"] = self.drop_turns_each_time
        self.config.save_config()
        yield event.plain_result(f"每次丢弃对话轮数已设为 {self.drop_turns_each_time}")

    @ctxlimit.command("setmargin")
    async def cmd_setmargin(self, event: AstrMessageEvent, margin: int):
        self.output_margin_tokens = max(0, int(margin))
        self.config["输出保留余量token"] = self.output_margin_tokens
        self.config.save_config()
        yield event.plain_result(f"输出保留余量已设为 {self.output_margin_tokens}")

    # === 关键钩子：在调用 LLM 之前裁剪上下文 ===
    @filter.on_llm_request(priority=100)  # 提高优先级，尽早处理
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.enabled:
            return

        # system 不丢
        system_prompt = req.system_prompt or ""
        contexts: List[Dict[str, Any]] = list(req.contexts or [])

        # 为计数构造 messages 视图
        def current_tokens() -> int:
            messages = _build_messages_for_count(req)
            return _count_tokens_openai_style(messages, system_prompt, self.count_model_hint)

        # 循环批量丢弃，直到满足“上下文+输出余量”不超过上限，或无可丢弃
        upper = max(512, int(self.max_context_tokens))
        margin = max(0, int(self.output_margin_tokens))

        messages_for_count = _build_messages_for_count(req)
        total = _count_tokens_openai_style(messages_for_count, system_prompt, self.count_model_hint)

        changed = False
        while total + margin > upper and contexts:
            before_len = len(contexts)
            contexts = _drop_old_turns(contexts, self.drop_turns_each_time)
            if len(contexts) == before_len:
                # 没有实际删除，避免死循环
                break
            req.contexts = contexts  # 写回
            messages_for_count = _build_messages_for_count(req)
            total = _count_tokens_openai_style(messages_for_count, system_prompt, self.count_model_hint)
            changed = True

        if changed:
            logger.info(
                f"[ctx_trimmer] 对 {event.unified_msg_origin} 进行了上下文裁剪。"
                f" tokens≈{total}, 上限={upper}, 余量={margin}, 剩余轮数={sum(1 for m in contexts if m.get('role')=='user')}"
            )

    # 可选：记录响应的真实 usage（若提供商返回），供调参参考
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            usage = getattr(resp, "raw_completion", {}) or {}
            if isinstance(usage, dict):
                meta = usage.get("usage") or {}
                if meta:
                    logger.debug(f"[ctx_trimmer] provider usage: {meta}")
        except Exception:
            pass
