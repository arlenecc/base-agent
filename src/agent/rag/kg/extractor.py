"""辅助模型抽取组件（spec 第九步，三种可切换方案）。

三元组抽取方案（配置项 ``rag_kg_extractor``，配置面板下拉切换）：
* ``llm_raw``      —— 自实现：辅助模型输出 ``entity<|#|>...`` / ``relation<|#|>...``
  行文本，本地正则解析（原实现，prompts 来自需求文档原文）；
* ``llm_pydantic`` —— 辅助模型输出严格 JSON，pydantic BaseModel 校验后转
  ExtractedEntity/Relation（结构化输出，字段非法即拒绝）；
* ``gliner``       —— 不基于 LLM：本地 GLiNER NER 模型抽实体，关系用同句
  共现启发式（无需辅助模型，首次使用需 ``pip install gliner`` 并下载模型）。

公共接口（三种方案一致，检索器/构建器无感知切换）：
* ``available`` —— 方案是否可用；
* ``extract_entities_relations(chunk_text, section_context)``
  → (List[ExtractedEntity], List[ExtractedRelation])；
* ``extract_keywords(query)`` → {"high_level_keywords": [...],
  "low_level_keywords": [...]}（hl/ll 双路召回输入）；
* ``summarize_descriptions(name, descs)`` —— 多描述归纳。

工厂 :func:`create_extractor` 按配置字符串构造对应实现。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from . import prompts

logger = logging.getLogger(__name__)

_ROW_DELIM = "<|#|>"
_COMPLETE_TAG = "<|COMPLETE|>"
_MAX_DESCRIPTIONS_FOR_SUMMARY = 12

_ENTITY_TYPES = {
    "Person", "Creature", "Organization", "Location", "Event", "Concept",
    "Method", "Content", "Data", "Artifact", "NaturalObject", "Other",
    "UNKNOWN",
}

# GLiNER NER 标签 → 本系统实体类型映射（gliner 输出自由标签，需归一化）。
GLINER_LABELS = [
    "person", "creature", "organization", "location", "event", "concept",
    "method", "creative work", "data", "product", "artifact", "natural object",
]
_GLINER_LABEL_MAP = {
    "person": "Person", "creature": "Creature", "organization": "Organization",
    "location": "Location", "event": "Event", "concept": "Concept",
    "method": "Method", "creative work": "Content", "data": "Data",
    "product": "Artifact", "artifact": "Artifact",
    "natural object": "NaturalObject",
}

# gliner 默认模型：多语种（含中文）NER。
GLINER_DEFAULT_MODEL = "urchade/gliner_multi"

# 三方案注册表（配置面板下拉 / 工厂共用）。
EXTRACTOR_SCHEMES = {
    "llm_raw": "LLM 行解析（自实现）",
    "llm_pydantic": "LLM 结构化（pydantic）",
    "gliner": "GLiNER 本地抽取（无需 LLM）",
}


class ExtractedEntity:
    __slots__ = ("name", "type", "description")

    def __init__(self, name: str, etype: str, description: str):
        self.name = name
        self.type = etype
        self.description = description


class ExtractedRelation:
    __slots__ = ("src", "dst", "keywords", "description")

    def __init__(self, src: str, dst: str, keywords: str, description: str):
        self.src = src
        self.dst = dst
        self.keywords = keywords
        self.description = description


def _normalize_type(etype: Any) -> str:
    t = str(etype or "").strip()
    return t if t in _ENTITY_TYPES else "Other"


# ---------------------------------------------------------------------------
# 公共基类
# ---------------------------------------------------------------------------

class BaseKGExtractor:
    """三种抽取方案的公共接口。子类按需覆盖。"""

    scheme = "base"

    def __init__(self, language: str = "中文"):
        self._language = language or "中文"
        self._unavailable_reason = "抽取器未初始化"

    @property
    def available(self) -> bool:
        raise NotImplementedError

    @property
    def unavailable_reason(self) -> str:
        """available=False 时给人看的失败原因（构建器据此报错）。"""
        return self._unavailable_reason

    def set_client(self, client: Any) -> None:
        """注入/替换底层客户端（LLM 方案用；GLiNER 忽略）。"""

    # -- 公共默认行为 ------------------------------------------------------

    def extract_entities_relations(
        self, chunk_text: str, section_context: str = "",
    ) -> tuple:
        """返回 (List[ExtractedEntity], List[ExtractedRelation])。"""
        raise NotImplementedError

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """hl/ll 关键词。默认回退：hl=[]，ll=[query]（保证实体检索有输入）。"""
        return {"high_level_keywords": [], "low_level_keywords": [query]}

    def summarize_descriptions(self, name: str, descriptions: List[str]) -> str:
        """多条描述 → 一段归纳。无 LLM 时拼接去重（保证不丢信息）。"""
        descs = [d.strip() for d in descriptions if d and d.strip()]
        if not descs:
            return ""
        if len(descs) == 1:
            return descs[0]
        seen, uniq = set(), []
        for d in descs[:_MAX_DESCRIPTIONS_FOR_SUMMARY]:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        return "；".join(uniq)


# ---------------------------------------------------------------------------
# 方案一：自实现 LLM 行解析（原 KGExtractor）
# ---------------------------------------------------------------------------

class KGExtractor(BaseKGExtractor):
    """辅助模型封装：抽取 / 关键词 / 归纳（``<|#|>`` 行解析）。"""

    scheme = "llm_raw"

    def __init__(self, llm_client: Any = None, language: str = "中文"):
        super().__init__(language)
        self._llm = llm_client

    @property
    def available(self) -> bool:
        if self._llm is None:
            self._unavailable_reason = (
                "辅助模型未配置：LLM 抽取方案需要辅助模型"
                "（配置面板「辅助模型」启用后重试）")
            return False
        # 客户端可声明自身可用性（如懒解析代理：辅助模型未加载时 False）。
        avail = getattr(self._llm, "available", None)
        if callable(avail):
            if not bool(avail()):
                self._unavailable_reason = "辅助模型未加载（请在配置面板加载辅助模型）"
                return False
            return True
        if isinstance(avail, bool):
            if not avail:
                self._unavailable_reason = "辅助模型未加载（请在配置面板加载辅助模型）"
                return False
        return True

    def set_client(self, client: Any) -> None:
        self._llm = client

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        if self._llm is None:
            return ""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # batch_chat 接口：多条独立 prompt 一次调用；这里单条复用同一路径，
        # MLX 后端退化 batch_chat 内部逐条收集，行为一致。
        try:
            outs = self._llm.batch_chat(
                [messages], tools=None, temperature=0.1, max_tokens=max_tokens,
                enable_thinking=False)
        except TypeError:
            # 客户端不支持 enable_thinking 形参时退回最简调用。
            outs = self._llm.batch_chat([messages], temperature=0.1,
                                        max_tokens=max_tokens)
        return (outs[0] if outs else "") or ""

    # ------------------------------------------------------------------
    # 实体 + 关系抽取
    # ------------------------------------------------------------------

    def extract_entities_relations(
        self, chunk_text: str, section_context: str = "",
    ) -> tuple:
        """返回 (List[ExtractedEntity], List[ExtractedRelation])。

        解析失败/无输出返回空列表，不影响整体构建。
        """
        user = prompts.EXTRACT_TASK.format(
            language=self._language,
            section_context=section_context or "（无）",
            input_text=chunk_text,
        )
        raw = self._chat(prompts.EXTRACT_SYSTEM, user, max_tokens=4096)
        return parse_extraction_output(raw)

    # ------------------------------------------------------------------
    # 查询关键词（hl/ll）
    # ------------------------------------------------------------------

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """返回 {"high_level_keywords": [...], "low_level_keywords": [...]}。

        失败时回退：hl=[]，ll=[query]（ll 兜底保证实体检索仍有输入）。
        """
        fallback = {"high_level_keywords": [], "low_level_keywords": [query]}
        if self._llm is None:
            return fallback
        raw = self._chat(
            "",  # 该 prompt 自带 Role，全部放 user 侧
            prompts.KEYWORDS_EXTRACTION.format(query=query, language=self._language),
            max_tokens=512)
        data = _parse_json_object(raw)
        if not isinstance(data, dict):
            logger.debug("KG: 关键词抽取输出不是 JSON，回退原始 query")
            return fallback
        hl = _string_list(data.get("high_level_keywords"))
        ll = _string_list(data.get("low_level_keywords"))
        if not hl and not ll:
            return fallback
        if not ll:
            ll = [query]
        return {"high_level_keywords": hl, "low_level_keywords": ll}

    # ------------------------------------------------------------------
    # 描述归纳
    # ------------------------------------------------------------------

    def summarize_descriptions(self, name: str, descriptions: List[str]) -> str:
        """多条描述 → 一段归纳（失败返回原样拼接，保证不丢信息）。"""
        descs = [d.strip() for d in descriptions if d and d.strip()]
        if not descs:
            return ""
        if len(descs) == 1:
            return descs[0]
        descs = descs[:_MAX_DESCRIPTIONS_FOR_SUMMARY]
        desc_list = "\n".join(
            json.dumps({"description": d}, ensure_ascii=False) for d in descs)
        merged = self._chat(
            "",
            prompts.SUMMARIZE_DESCRIPTIONS.format(
                description_name=name, description_list=desc_list),
            max_tokens=1024).strip()
        merged = merged.strip("`").strip()
        return merged if merged else " ".join(descs)


# ---------------------------------------------------------------------------
# 方案二：LLM 结构化输出 + pydantic 校验
# ---------------------------------------------------------------------------

def _build_pydantic_models():
    """延迟导入 pydantic 并构建抽取结果模型（pydantic 未装时返回 None）。"""
    try:
        from pydantic import BaseModel, Field, field_validator
    except ImportError:  # pragma: no cover - 环境缺依赖
        return None

    class PydanticEntity(BaseModel):
        name: str = Field(min_length=1)
        type: str = "Other"
        description: str = ""

        @field_validator("type")
        @classmethod
        def _valid_type(cls, v: str) -> str:
            return _normalize_type(v)

    class PydanticRelation(BaseModel):
        src: str = Field(min_length=1)
        dst: str = Field(min_length=1)
        keywords: str = ""
        description: str = ""

    class PydanticExtraction(BaseModel):
        entities: List[PydanticEntity] = Field(default_factory=list)
        relations: List[PydanticRelation] = Field(default_factory=list)

    return PydanticExtraction


class PydanticLLMExtractor(KGExtractor):
    """LLM 结构化抽取：prompt 要求严格 JSON，pydantic 校验后转结果对象。

    与 llm_raw 的差异：字段缺失/类型不符的行会被 pydantic 整体拒绝
    （而不是宽松切列），适合对数据质量要求更高的构建；模型输出非法
    JSON 时记日志并返回空结果（单片失败不中断整体构建）。
    """

    scheme = "llm_pydantic"

    def __init__(self, llm_client: Any = None, language: str = "中文"):
        super().__init__(llm_client, language)
        self._models = _build_pydantic_models()
        if self._models is None:
            logger.warning("KG: pydantic 未安装，llm_pydantic 方案不可用")

    @property
    def available(self) -> bool:
        if self._models is None:
            self._unavailable_reason = (
                "pydantic 未安装：llm_pydantic 方案需要 pydantic"
                "（pip install pydantic）")
            return False
        return super().available

    def extract_entities_relations(
        self, chunk_text: str, section_context: str = "",
    ) -> tuple:
        if self._models is None:
            return [], []
        user = prompts.PYDANTIC_EXTRACT_TASK.format(
            language=self._language,
            section_context=section_context or "（无）",
            input_text=chunk_text,
        )
        raw = self._chat(prompts.PYDANTIC_EXTRACT_SYSTEM, user, max_tokens=4096)
        data = _parse_json_object(raw)
        if not isinstance(data, dict):
            logger.debug("KG: pydantic 方案输出不是 JSON 对象，本片返回空结果")
            return [], []
        try:
            parsed = self._models.model_validate(data)
        except Exception as e:  # pydantic ValidationError 等
            logger.debug("KG: pydantic 校验失败: %s", e)
            return [], []
        entities = [ExtractedEntity(en.name.strip(), en.type, en.description)
                    for en in parsed.entities if en.name.strip()]
        relations = [ExtractedRelation(r.src.strip(), r.dst.strip(),
                                       r.keywords, r.description)
                     for r in parsed.relations
                     if r.src.strip() and r.dst.strip()]
        # 无向关系去重（与 llm_raw 的 parse_extraction_output 行为一致）。
        return entities, _dedup_relations(relations)


# ---------------------------------------------------------------------------
# 方案三：GLiNER 本地 NER（不基于 LLM）
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_MAX_REL_PAIRS_PER_SENT = 8


class GLiNERExtractor(BaseKGExtractor):
    """本地 GLiNER 抽取：实体用 NER，关系用同句共现启发式，无 LLM 依赖。

    * 实体：``GLiNER.predict_entities(text, labels)``，标签映射回本系统
      实体类型；threshold 之下的弱召回丢弃；
    * 关系：同一句内共现的实体两两连边（每句最多 8 对），keywords 用
      实体类型对（如 ``Person-Organization``），description 用原句；
    * 查询关键词：GLiNER 抽 query 里的实体作 ll，hl 置空；
    * 描述归纳：无 LLM，去重拼接（继承基类默认实现）。

    模型懒加载：首次抽取时才 ``GLiNER.from_pretrained``（需要网络下载
    或本地缓存）；未安装 gliner 包时 available=False 并给出安装提示。
    """

    scheme = "gliner"

    def __init__(self, model_name: str = GLINER_DEFAULT_MODEL,
                 language: str = "中文", threshold: float = 0.4):
        super().__init__(language)
        self._model_name = model_name or GLINER_DEFAULT_MODEL
        self._threshold = float(threshold)
        self._model: Any = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        try:
            import gliner  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "gliner 未安装：GLiNER 方案需要 gliner 包"
                "（pip install gliner），首次使用还需下载 NER 模型权重")
            return False
        if self._load_failed:
            self._unavailable_reason = (
                f"GLiNER 模型 {self._model_name} 加载失败（检查网络或本地缓存）")
            return False
        return True

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError as e:
            self._load_failed = True
            raise RuntimeError(
                "gliner 未安装：pip install gliner（GLiNER 方案不依赖 LLM，"
                "但需要本地 NER 模型包）") from e
        try:
            logger.info("KG: 加载 GLiNER 模型 %s ...", self._model_name)
            self._model = GLiNER.from_pretrained(self._model_name)
        except Exception as e:
            self._load_failed = True
            raise RuntimeError(
                f"GLiNER 模型加载失败: {e}（模型 {self._model_name} 需要网络"
                "下载或已存在本地 HF 缓存）") from e
        return self._model

    # -- 实体 + 关系 -------------------------------------------------------

    def extract_entities_relations(
        self, chunk_text: str, section_context: str = "",
    ) -> tuple:
        model = self._ensure_model()
        # GLiNER 无「上下文」概念：section_context 拼在正文前一起抽，
        # 去重逻辑会合并重复实体。
        full_text = (f"{section_context}\n{chunk_text}"
                     if section_context else chunk_text)
        labels = [lb for lb in GLINER_LABELS]
        try:
            spans = model.predict_entities(full_text, labels,
                                           threshold=self._threshold)
        except TypeError:
            # 旧版 gliner 没有 threshold 形参。
            spans = model.predict_entities(full_text, labels)
        entities: List[ExtractedEntity] = []
        ents_in_order: List[tuple] = []  # (text, type) 按出现顺序
        seen_names = set()
        for sp in spans:
            name = str(sp.get("text", "") or "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            etype = _GLINER_LABEL_MAP.get(
                str(sp.get("label", "")).strip().lower(), "Other")
            entities.append(ExtractedEntity(name, etype, ""))
            ents_in_order.append((name, etype))

        # 同句共现 → 启发式关系
        relations: List[ExtractedRelation] = []
        seen_pairs = set()
        for sent in _SENT_SPLIT_RE.split(full_text):
            present = [(n, t) for n, t in ents_in_order if n in sent]
            if len(present) < 2:
                continue
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    if len(relations) >= 40:
                        break
                    src, dst = present[i], present[j]
                    pk = "\x1f".join(sorted([src[0].lower(), dst[0].lower()]))
                    if pk in seen_pairs:
                        continue
                    seen_pairs.add(pk)
                    relations.append(ExtractedRelation(
                        src[0], dst[0],
                        f"{src[1]}-{dst[1]}",
                        sent.strip()[:200]))
        return entities, _dedup_relations(relations)

    # -- 查询关键词 --------------------------------------------------------

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        fallback = {"high_level_keywords": [], "low_level_keywords": [query]}
        try:
            model = self._ensure_model()
        except RuntimeError as e:
            logger.warning("KG: GLiNER 关键词抽取不可用，回退原始 query: %s", e)
            return fallback
        try:
            spans = model.predict_entities(query, GLINER_LABELS,
                                           threshold=self._threshold)
        except TypeError:
            spans = model.predict_entities(query, GLINER_LABELS)
        ll = []
        for sp in spans:
            name = str(sp.get("text", "") or "").strip()
            if name and name not in ll:
                ll.append(name)
        if not ll:
            return fallback
        return {"high_level_keywords": [], "low_level_keywords": ll[:10]}


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def create_extractor(scheme: str, llm_client: Any = None,
                     language: str = "中文") -> BaseKGExtractor:
    """按配置字符串构造抽取方案。

    scheme ∈ {llm_raw, llm_pydantic, gliner}；未知值回退 llm_raw 并告警。
    """
    scheme = str(scheme or "llm_raw").lower()
    if scheme == "llm_pydantic":
        return PydanticLLMExtractor(llm_client, language)
    if scheme == "gliner":
        return GLiNERExtractor(language=language)
    if scheme != "llm_raw":
        logger.warning("KG: 未知抽取方案 %r，回退 llm_raw", scheme)
    return KGExtractor(llm_client, language)


# ---------------------------------------------------------------------------
# 输出解析（llm_raw 方案）
# ---------------------------------------------------------------------------

def parse_extraction_output(raw: str) -> tuple:
    """解析 ``entity<|#|>...`` / ``relation<|#|>...`` 行输出。"""
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []
    if not raw:
        return entities, relations
    # 掐掉 <|COMPLETE|> 之后与 <think> 段（辅助模型可能带思考前缀）。
    if _COMPLETE_TAG in raw:
        raw = raw.split(_COMPLETE_TAG, 1)[0]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    for line in raw.splitlines():
        line = line.strip()
        if not line or _ROW_DELIM not in line:
            continue
        parts = [p.strip() for p in line.split(_ROW_DELIM)]
        tag = parts[0].lower()
        try:
            if tag == "entity" and len(parts) >= 4:
                name, etype, desc = parts[1], parts[2], parts[3]
                if not name:
                    continue
                if etype not in _ENTITY_TYPES:
                    etype = "Other"
                entities.append(ExtractedEntity(name, etype, desc))
            elif tag == "relation" and len(parts) >= 5:
                src, dst, kws, desc = parts[1], parts[2], parts[3], parts[4]
                if not src or not dst:
                    continue
                relations.append(ExtractedRelation(src, dst, kws, desc))
        except IndexError:  # pragma: no cover - 防御畸形行
            continue
    # 关系去重（无向归一化）
    return entities, _dedup_relations(relations)


def _dedup_relations(relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
    seen = set()
    deduped: List[ExtractedRelation] = []
    for r in relations:
        pk = "\x1f".join(sorted([r.src.lower(), r.dst.lower()]))
        if pk in seen:
            continue
        seen.add(pk)
        deduped.append(r)
    return deduped


def _parse_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
