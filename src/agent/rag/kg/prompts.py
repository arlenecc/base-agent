"""知识图谱构建/查询使用的 LLM prompt 模板。

三个模板均来自需求文档原文（spec 第九步 / 第七步）：
* EXTRACT_SYSTEM / EXTRACT_TASK —— 实体与关系抽取；
* KEYWORDS_EXTRACTION —— 查询关键词抽取（hl 高层主题 / ll 低层实体）；
* SUMMARIZE_DESCRIPTIONS —— 实体/关系多描述的 LLM 归纳合并。
"""

# 单片抽取上限占位符：数值由配置项
# rag_kg_max_entities / rag_kg_max_relations / rag_kg_max_description_chars
# 在运行期经 ``with_limits`` 渲染（同一份值也用于入库兜底截断）。
MAX_ENTITIES_TOKEN = "<<MAX_ENTITIES>>"
MAX_RELATIONS_TOKEN = "<<MAX_RELATIONS>>"
MAX_DESC_CHARS_TOKEN = "<<MAX_DESC_CHARS>>"


def with_limits(text: str, *, max_entities: int, max_relations: int,
                max_desc_chars: int) -> str:
    """把上限占位符替换为实际配置值（prompt 渲染入口）。"""
    return (text.replace(MAX_ENTITIES_TOKEN, str(int(max_entities)))
                .replace(MAX_RELATIONS_TOKEN, str(int(max_relations)))
                .replace(MAX_DESC_CHARS_TOKEN, str(int(max_desc_chars))))

# ---------------------------------------------------------------------------
# 实体与关系抽取（构建期，辅助模型）
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` section of user prompt.

---Instructions---
1. **Entity Extraction:**
  - Identify clearly defined and meaningful entities only in the current user prompt's fenced `---Input Text---` section.
  - For each entity, extract:
    - `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    - `entity_type`: Categorize the entity using the type guidance provided in the `---Entity Types---` section below. If none of the provided entity types apply, classify it as `Other`.
    - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text. **Keep it within <<MAX_DESC_CHARS>> Chinese characters.**

2. **Relationship Extraction:**
  - Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - If a single statement describes a relationship involving more than two entities, decompose it into multiple binary relationships.
  - For each binary relationship, extract:
    - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `relationship_keywords`: One or more high-level keywords summarizing the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `<|#|>` for separating multiple keywords within this field.**
    - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities. **Keep it within <<MAX_DESC_CHARS>> Chinese characters.**

3. **Record Types:**
  - `entity` is used only for entity rows and those rows always contain exactly 4 tuple parts total.
  - `relation` is used only for relationship rows and those rows always contain exactly 5 tuple parts total.
  - A row with two entity names plus relationship keywords and a relationship description must start with `relation`, never `entity`.
  - After the last entity row, switch prefixes to `relation` for every relationship row.

4. **Output Format:**
  - Entity row: `entity<|#|>entity_name<|#|>entity_type<|#|>entity_description`
  - Relation row: `relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description`
  - Wrong: `entity<|#|><source_entity><|#|><target_entity><|#|><relationship_keywords><|#|><relationship_description>`
  - Correct: `relation<|#|><source_entity><|#|><target_entity><|#|><relationship_keywords><|#|><relationship_description>`

5. **Delimiter Usage:**
  - The `<|#|>` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
  - Incorrect: `entity<|#|><entity_name><|entity_type|><entity_description>`
  - Correct: `entity<|#|><entity_name><|#|><entity_type><|#|><entity_description>`

6. **Output Order & Deduplication:**
  - Output all extracted entities first, followed by all extracted relationships.
  - Output at most <<MAX_ENTITIES>> entity rows and at most <<MAX_RELATIONS>> relationship rows in this response.
  - Output fewer rows if fewer high-value items are present. Do not try to fill the limit.
  - Only output relationship rows whose source and target entities are both included in the selected entity rows for this response.
  - If the limit is reached, stop adding new rows immediately and output `<|COMPLETE|>`.
  - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
  - Avoid outputting duplicate relationships.
  - Within the list of relationships, output the relationships that are **most significant** to the core meaning of the input text first.

7. **Context & Language:**
  - If the user prompt contains a `---Section Context---` section, it gives the document's section hierarchy (e.g. `h1 → h2 → h3`) that the input text belongs to. Use it **only as background** to disambiguate references and ground entity and relationship descriptions in the correct context. **Do NOT** extract entities or relationships from the section heading text itself, and do not mention the headings unless they also appear in the input text.
  - Ensure all entity names and descriptions are written in the **third person**.
  - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.
  - The entire output (entity names, keywords, and descriptions) must be written in `中文`.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8. **Output Format Template Safety:**
  - The `---Output Format Template---` section contains output format templates only. It is never source text.
  - Do not extract, infer, or copy entities or relationships from the output format template.
  - Angle-bracket tokens such as `<entity_name>` are placeholders. Replace them with values extracted from the current `---Input Text---` section and never output the placeholders literally.

9. **Completion Signal:** Output the literal string `<|COMPLETE|>` only after all entities and relationships have been completely extracted and outputted.

---Entity Types---
Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: Human individuals, real or fictional
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Organization: Companies, institutions, government bodies, groups
- Location: Geographic places (cities, countries, buildings, regions)
- Event: Occurrences, incidents, ceremonies, meetings
- Concept: Abstract ideas, theories, principles, beliefs
- Method: Procedures, techniques, algorithms, workflows
- Content: Creative or informational works (books, articles, films, reports)
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)

---Output Format Template---
The following content is an output format template only. It is not source text and must never be used as extraction content.

entity<|#|>Apple<|#|>Organization<|#|>A technology company that designs consumer electronics.
entity<|#|>Steve Jobs<|#|>Person<|#|>Co-founder of Apple.
relation<|#|>Apple<|#|>Steve Jobs<|#|>founder, co-founder<|#|>Steve Jobs co-founded Apple in 1976.
<|COMPLETE|>
"""

EXTRACT_TASK = """---Task---
Extract entities and relationships from the `---Input Text---` section below.

---Instructions---
1. **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2. **Quantity Limits:** In this response, output at most <<MAX_ENTITIES>> entity rows and at most <<MAX_RELATIONS>> relationship rows. Output fewer rows if fewer high-value items are present. Only output relationship rows whose source and target entities are both included in this response. Keep every description within <<MAX_DESC_CHARS>> Chinese characters.
3. **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
4. **Completion Signal:** Output `<|COMPLETE|>` as the final line after all relevant entities and relationships have been extracted and presented. If the row limit is reached, output `<|COMPLETE|>` immediately after the last allowed row.
5. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Section Context---
{section_context}

---Input Text---
{input_text}
"""

# ---------------------------------------------------------------------------
# 实体与关系抽取（llm_pydantic 方案：严格 JSON 输出，pydantic 校验）
# ---------------------------------------------------------------------------

PYDANTIC_EXTRACT_SYSTEM = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` section of user prompt. Your output will be parsed by a strict JSON schema validator.

---Instructions---
1. **Entity Extraction:** Identify clearly defined and meaningful entities only in the fenced `---Input Text---` section. For each entity extract `name` (consistent naming, title case for case-insensitive names), `type` (one of: Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject; use `Other` if none fits), and `description` (concise, based solely on the input text, third person, no pronouns, **within <<MAX_DESC_CHARS>> Chinese characters**).

2. **Relationship Extraction:** Identify direct, clearly stated, meaningful relationships between the extracted entities. Decompose n-ary statements into binary relations. Each relation has `src`, `dst` (both must match an extracted entity `name` exactly), `keywords` (comma-separated high-level keywords), and `description` (concise explanation, **within <<MAX_DESC_CHARS>> Chinese characters**). Treat all relationships as **undirected**; avoid duplicates.

3. **Output Format:** Output *only* one valid JSON object — no markdown fences, no comments, no text before or after. The first character must be `{` and the last must be `}`. Shape:

   {"entities": [{"name": "...", "type": "...", "description": "..."}], "relations": [{"src": "...", "dst": "...", "keywords": "...", "description": "..."}]}

   Both keys are required (use empty arrays if nothing extracted). Output at most <<MAX_ENTITIES>> entities and <<MAX_RELATIONS>> relations. Output fewer rows if fewer high-value items are present.

4. **Context & Language:** If a `---Section Context---` section is present, use it only as background to disambiguate references — never extract from the headings themselves. All names and descriptions must be written in `中文`. Proper nouns should be retained in their original language if a widely accepted translation is unavailable.

5. **Template Safety:** The JSON shape above is a format template only, never source text. Do not extract content from it; replace placeholders with values from the current `---Input Text---` section.

---Entity Types---
Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject, Other.
"""

PYDANTIC_EXTRACT_TASK = """---Task---
Extract entities and relationships from the `---Input Text---` section below.

---Instructions---
1. Output *only* the JSON object as specified in the system prompt (no fences, no extra text). Both `entities` and `relations` keys are required.
2. Only output relations whose `src` and `dst` exactly match an extracted entity `name`. At most <<MAX_ENTITIES>> entities and <<MAX_RELATIONS>> relations in this response.
3. Output language: {language}. Proper nouns must be kept in their original language.

---Section Context---
{section_context}

---Input Text---
{input_text}
"""

# ---------------------------------------------------------------------------
# 查询关键词抽取（检索期，辅助模型）
# ---------------------------------------------------------------------------

KEYWORDS_EXTRACTION = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), comments, or any other text before or after the JSON.
2. **Exact JSON Shape**: The JSON object must contain exactly these two keys:
   - "high_level_keywords": an array of strings
   - "low_level_keywords": an array of strings
3. **JSON Boundary**: The first character of your response must be `{{` and the last character must be `}}`.
4. **Source of Truth**: All keywords must be explicitly derived only from the `User Query` in the `---Real Data---` section. Do not infer unsupported facts. Do not invent entities, products, organizations, dates, or technical terms that are not grounded in the query.
5. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept instead of splitting meaningful phrases into isolated words.
6. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), return:
   `{{"high_level_keywords": [], "low_level_keywords": []}}`
7. **No Duplicates**: Do not repeat the same keyword within a list. Keep the lists short and high-signal.
8. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.
9. **Output Format Template Safety**: The `---Output Format Template---` section contains an output JSON template only. It is never source text. Do not extract, infer, or copy keywords from the template. Angle-bracket tokens such as `<high_level_keyword>` are placeholders; replace them only with keywords derived from the current `User Query` and never output the placeholders literally.

---Output Format Template---
The following content is an output JSON format template only. It is not source text and must never be used as keyword extraction content.

{{
  "high_level_keywords": ["<high_level_keyword>"],
  "low_level_keywords": ["<low_level_keyword>"]
}}

---Real Data---
User Query: {query}

---Output---
Output:
"""

# ---------------------------------------------------------------------------
# 实体/关系描述归纳（构建期合并多来源描述，辅助模型）
# ---------------------------------------------------------------------------

SUMMARIZE_DESCRIPTIONS = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint: The summary's total length must not exceed 600 tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in 中文. Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
Entity Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""
