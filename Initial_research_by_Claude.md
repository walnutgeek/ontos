# Initial Research by Claude: Supplementary Findings

This document provides additional research findings to complement the comprehensive survey in `Initial_research_of_sources_prplx_labs.md`. Focus is on notable contributions, emerging paradigms, and technical foundations that are underrepresented or absent from the primary research.

---

## 1. Advanced Reasoning Frameworks Beyond GraphRAG

### Graph-Constrained Reasoning (GCR)
A novel framework that eliminates hallucinations by integrating KG structure directly into the LLM decoding process through **KG-Trie**, a trie-based index encoding KG reasoning paths. Unlike post-hoc retrieval, GCR constrains decoding to ensure faithful KG-grounded reasoning.
- Source: [OpenReview - Graph-constrained Reasoning](https://openreview.net/forum?id=6embY8aclt)

### REKG-MCTS
Uses Monte Carlo Tree Search for reasoning over knowledge graphs. The LLM agent explores multiple reasoning paths through MCTS before selecting the optimal path for final answer generation.
- Source: [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.484.pdf)

### Chain-of-Knowledge (CoK)
Augments LLMs by dynamically incorporating grounding information from **heterogeneous sources** using an adaptive query generator that produces queries for SPARQL, SQL, and natural language. Progressively corrects rationales using preceding corrections.
- Source: [OpenReview - Chain-of-Knowledge](https://openreview.net/forum?id=cPgh4gWZlz)

### Tree of Thoughts (ToT) and Graph of Thoughts (GoT)
Extends Chain-of-Thought beyond linear reasoning:
- **ToT**: Branches into multiple pathways like decision trees, enabling exploration of diverse reasoning paths with lookahead and backtracking
- **GoT**: Models LLM-generated information as arbitrary graphs where units are vertices connected by interdependency edges
- Source: [Medium - Advanced Reasoning Frameworks](https://medium.com/@dewanshsinha71/advanced-reasoning-frameworks-in-large-language-models-chain-tree-and-graph-of-thoughts-bafbfd028575)

---

## 2. Causal Reasoning and Knowledge Graphs

### Causal Knowledge Graph Extraction
Fujitsu has developed technology that automatically extracts causal relationships from documents without training data, leveraging LLMs. Applications include:
- Applying KGs during pre-training
- Using KGs for LLM inference
- Utilizing KGs to understand and interpret LLM reasoning processes
- Source: [Fujitsu Causal Knowledge Graph Whitepaper](https://www.fujitsu.com/global/documents/about/research/article/202410-causal-knowledge-graph/202410_White-Paper-Casual-Knowledge-Graph_EN.pdf)

### Causal Graphs Meet Thoughts
Framework emphasizing two principles:
1. Prioritizing causal edges within knowledge graphs
2. Aligning retrieval with the LLM's chain-of-thought to dynamically guide each reasoning step

Achieves up to **10% absolute gain** in diagnostic accuracy on medical QA tasks.
- Source: [arXiv - Causal Graphs Meet Thoughts](https://arxiv.org/abs/2501.14892)

### Grounding LLM Reasoning with Knowledge Graphs
Recent framework (Feb 2025) that integrates LLM reasoning with KGs by linking each step to graph-structured data. Incorporates CoT, ToT, and GoT strategies. Achieves **26.5%+ improvement** over CoT baselines.
- Source: [arXiv 2502.13247](https://arxiv.org/abs/2502.13247)

---

## 3. Graph Neural Networks + LLM Integration

### GNN-RAG Framework
Combines GNN's knowledge graph processing abilities with LLM's language abilities:
- GNN acts as dense subgraph reasoner to extract useful graph information
- Extracted paths are verbalized and given as input to LLMs (ChatGPT, Llama)
- LLM leverages NLP ability for ultimate KGQA
- Source: [LLMs and GNNs for RAG - Substack](https://bdtechtalks.substack.com/p/llms-and-gnns-are-a-killer-combo)

### LinguGKD (Linguistic Graph Knowledge Distillation)
Uses LLMs as teacher models and GNNs as student models for knowledge distillation:
- Boosts student GNN's predictive accuracy and convergence rate
- Distilled GNN achieves superior inference speed with fewer computing demands
- Source: [arXiv - LLM Meets GNN in Knowledge Distillation](https://arxiv.org/abs/2402.05894)

### Graph Neural Prompting (GNP)
Plug-and-play method to assist pre-trained LLMs in learning from KGs, including:
- Standard GNN encoder
- Cross-modality pooling module
- Domain projector
- Self-supervised link prediction objective
- Source: [arXiv 2309.15427](https://arxiv.org/abs/2309.15427)

### Key Frameworks
- **PyTorch Geometric (PyG)** and **Deep Graph Library (DGL)** for robust GNN implementation in hybrid LLM systems

---

## 4. Commonsense Knowledge Bases (Underrepresented in Main Survey)

### ConceptNet
The most widely-used general commonsense knowledge graph:
- 250,000+ elements of commonsense knowledge
- Sourced from Wiktionary, Open Mind Common Sense, WordNet, Verbosity
- Represents common concepts and relationships between them
- Source: [ResearchGate - ConceptNet](https://www.researchgate.net/publication/2930180_ConceptNet-A_Practical_Commonsense_Reasoning_Tool-Kit)

### ATOMIC
Commonsense knowledge base focusing on inferential if-then knowledge:
- Typed if-then relations with variables (e.g., "if X pays Y a compliment, then Y will likely return the compliment")
- Covers social and physical commonsense
- **ATOMIC 2020** extends coverage significantly
- Source: [arXiv - ATOMIC](https://arxiv.org/pdf/1811.00146)

### COMET (Commonsense Transformers)
Learns to generate commonsense descriptions in natural language:
- Up to **77.5% precision** (ATOMIC) and **91.7% precision** (ConceptNet) at top-1
- Generates novel knowledge rated high quality by humans
- Source: [arXiv 1906.05317](https://arxiv.org/abs/1906.05317)

### BRIGHT Framework (2025)
Generates novel commonsense knowledge with:
- Up to **397K** new knowledge entries
- GPT-4 acceptance rates of **90.51%** (ATOMIC) and **85.59%** (ConceptNet)
- Source: [ScienceDirect - Amplifying commonsense knowledge](https://www.sciencedirect.com/science/article/abs/pii/S030645732500010X)

---

## 5. Knowledge Graph Embeddings (Technical Foundation)

### Core Models

| Model | Approach | Key Innovation |
|-------|----------|----------------|
| **TransE** | Translation-based | First translation model; relation as translation between entity vectors |
| **ComplEx** | Complex numbers | Handles asymmetric relations via complex space |
| **RotatE** | Rotational | Models relations as rotations in complex space; handles symmetric, antisymmetric, inverse, and composition patterns |
| **ConvE** | CNN-based | First to use 2D convolution for link prediction |
| **R-GCN** | GNN-based | Relational graph convolution for multi-relational data |

### RotatE Details
Uses Euler's identity (e^(iθ) = cos θ + i sin θ) to model relations as rotations. Symmetric relations modeled as 180-degree rotations.
- Source: [OpenReview - RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ)
- Implementation: [GitHub - DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

---

## 6. Knowledge Editing in LLMs (Missing from Main Survey)

### ROME (Rank-One Model Editing)
Modifies individual facts within GPT by treating MLP modules as key-value stores:
- Key encodes subject, value encodes knowledge about subject
- Makes localized weight changes to update specific facts
- Example: Teaching GPT-J that "Eiffel Tower is located in Rome"
- Source: [ROME - MIT](https://rome.baulab.info/)

### MEMIT (Mass-Editing Memory in a Transformer)
Scales ROME to thousands of simultaneous edits:
- Spreads memory updates across multiple MLP layers
- Remains stable after 40+ edits (vs. ROME degrading after 10)
- Source: [MEMIT - MIT](https://memit.baulab.info/)

### Limitations
Multi-hop question performance degrades catastrophically after editing (30.2% → 4.9% with Vicuna-7B).
- Source: [MQuAKE - arXiv](https://arxiv.org/abs/2305.14795)

### EasyEdit Framework
ACL 2024 framework supporting ROME, MEMIT, MEND with evaluation metrics for Reliability, Generalization, Locality, and Portability.
- Source: [GitHub - zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)

---

## 7. Self-RAG and Adaptive Retrieval

### Self-RAG Framework
Trains LLMs to adaptively retrieve passages on-demand using **reflection tokens**:
- **Retrieval Decision Tokens**: [Retrieve] or [No Retrieval]
- Model decides at each step if it needs additional evidence
- Critiques retrieved passages via reflection tokens
- Outperforms ChatGPT and retrieval-augmented Llama2-chat on QA, reasoning, and fact verification
- Source: [Self-RAG Project Page](https://selfrag.github.io/)

### Auto-RAG
Autonomous iterative retrieval that adjusts iteration count based on question difficulty and utility of retrieved knowledge, without human intervention.
- Source: [Learn Prompting - Auto-RAG](https://learnprompting.org/docs/retrieval_augmented_generation/auto-rag)

---

## 8. Table Augmented Generation (TAG)

Emerging paradigm for structured data that differs from standard RAG:
- Leverages SQL-based querying to fetch rows and columns directly from relational tables
- Augments results with AI insights (anomaly detection, trends, predictions)
- Suited for enterprise scenarios requiring accuracy and compliance (finance, retail, insurance)
- Source: [GigaSpaces - From RAG to TAG](https://www.gigaspaces.com/blog/from-rag-to-tag)

### Text-to-SQL Integration
- SQL enables expressive analytics: aggregations, joins, sorting, filtering
- LLM converts natural language to SQL as "program synthesis cheat code"
- Combining Text-to-SQL with semantic search provides best of both worlds
- Source: [LlamaIndex - Combining Text-to-SQL with Semantic Search](https://www.llamaindex.ai/blog/combining-text-to-sql-with-semantic-search-for-retrieval-augmented-generation-c60af30ec3b)

---

## 9. Multimodal Knowledge Graphs (Underrepresented)

### MR-MKG (Multimodal Reasoning with MMKG)
Expands multimodal knowledge of LLMs by learning from multimodal knowledge graphs:
- **1.95%** accuracy increase over state-of-the-art
- **10.4%** improvement in Hits@1 metric
- Source: [arXiv 2406.02030](https://arxiv.org/abs/2406.02030)

### VaLiK (Vision-align-to-Language Knowledge Graph)
Text-free approach to MMKG construction:
- Uses pretrained Vision-Language models based on Chain-of-Experts
- Converts visual inputs to image-specific textual descriptions via cross-modal feature alignment
- Source: [arXiv 2503.12972](https://arxiv.org/abs/2503.12972)

### GraphVis
Conserves intricate graph structure through visual modality to enhance KG comprehension with Large Vision Language Models (LVLMs).
- Source: [NeurIPS 2024 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7cb04f510593c9ba30da398f5e0a7e7b-Paper-Conference.pdf)

---

## 10. Major Public Knowledge Graphs (Context)

### Comparison of Open Knowledge Graphs

| KG | Size | Source | Strengths |
|----|------|--------|-----------|
| **Wikidata** | 100M+ items | Community-edited | Timeliness, continuous SPARQL access |
| **DBpedia** | Extracted from Wikipedia | Automated | DBpedia LIVE for updates, strong linked data |
| **YAGO** | 10M+ entities, 120M+ facts | Wikidata + Schema.org | 95%+ accuracy, linked to SUMO ontology |
| **Freebase** | Deprecated (migrated to Wikidata) | Google | Historical significance |

Key insight: These KGs are **not interchangeable**—each has strengths for different application domains.
- Source: [Springer - One Knowledge Graph to Rule Them All?](https://link.springer.com/chapter/10.1007/978-3-319-67190-1_33)

---

## 11. Natural Language to SPARQL

### SPARQL-LLM (2024-2025)
Open-source, triplestore-agnostic approach for real-time SPARQL generation:
- **24% F1 Score increase** over state-of-the-art
- **36x faster** than other systems
- Maximum **$0.01 per question**
- Source: [arXiv 2512.14277](https://arxiv.org/abs/2512.14277)

### Commercial Solutions
**Lymba NL2Query™**: Converts plain English to SPARQL, works with Stardog, AnzoGraph, MarkLogic, AllegroGraph, Oracle.
- Source: [Lymba NL2Query](https://www.lymba.com/nl2query)

---

## 12. Hallucination Detection and Mitigation

### REFIND Framework
Retrieval-augmented Factuality hallucINation Detection:
- Quantifies context sensitivity of each token
- Calculates Context Sensitivity Ratio (CSR)
- High CSR tokens identified as likely hallucinations
- Source: [arXiv 2502.13622](https://arxiv.org/abs/2502.13622)

### SAFE (Search-Augmented Factuality Evaluator)
Uses LLM as agent to iteratively issue Google Search queries:
- **72% agreement** rate with humans
- **76% win rate** over humans when they disagree
- **20x cheaper** than human annotators
- Source: [Lil'Log - Extrinsic Hallucinations](https://lilianweng.github.io/posts/2024-07-07-hallucination/)

### FACTSCORE
Fine-grained factual metric for long-form text:
1. Decomposes content into atomic facts
2. Computes percentage supported by reliable sources

### Combined Approach Effectiveness
Stanford 2024 study: Combining RAG, RLHF, and guardrails led to **96% reduction** in hallucinations vs. baseline.
- Source: [MDPI - Hallucination Mitigation for RAG-LLMs](https://www.mdpi.com/2227-7390/13/5/856)

---

## 13. Question Answering Over Knowledge Bases (KBQA)

### Two Main Paradigms

1. **Information Retrieval (IR)-based**: Binary classification over candidate answers using distributed representations
2. **Neural Semantic Parsing (NSP)**: Translates questions to executable logical forms

### NSQA (Neuro-Symbolic Question Answering)
Modular system achieving SOTA on QALD-9 and LC-QuAD 1.0:
- Uses Abstract Meaning Representation (AMR) parses
- Graph transformation to convert AMR to candidate logical queries
- Pipeline integrating reusable modules
- Source: [arXiv 2012.01707](https://arxiv.org/abs/2012.01707v1)

### Key Benchmark Datasets
- **WebQuestions** (Freebase, EMNLP 2013)
- **LC-QuAD** (Complex questions, ISWC 2017)
- **QALD-9** (Question Answering over Linked Data)
- **HotpotQA** (Multi-hop reasoning)

### Resource
Comprehensive paper list: [GitHub - Awesome-KBQA](https://github.com/RUCAIBox/Awesome-KBQA)

---

## 14. Open Information Extraction for KG Construction

### KGGen
Open-source package using LMs for high-quality KG extraction:
- LM-based extractor predicts subject-predicate-object triples
- Iterative LM-based clustering refines raw graph
- **18% improvement** over leading extractors on benchmarks
- Source: [arXiv - KGGen](https://arxiv.org/abs/2502.09956)

### LOKE (Linked Open Knowledge Extraction)
Combines prompt engineering with entity linking:
- LOKE-GPT outperforms AllenAI's OpenIE 4
- Addresses inadequate alignment of OpenIE results with existing KGs
- Source: [arXiv 2311.09366](https://arxiv.org/abs/2311.09366)

### Plumber Framework
Brings together 40 reusable components for KG completion:
- Coreference resolution, entity linking, relation extraction
- Offers 432 distinct pipeline configurations
- Source: [PMC - Information extraction pipelines](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823264/)

---

## 15. IBM's LLM Store for Wikidata

### KIF Plugin Architecture
Plugin for Wikidata-based knowledge integration that uses LLMs as knowledge sources:
- Instead of consulting static KB, synthesizes Wikidata-like statements on-the-fly
- Achieved **90.83% macro F1-score** in LM-KBC Challenge @ ISWC 2024
- Source: [IBM Research - LLM Store](https://research.ibm.com/publications/llm-store-a-kif-plugin-for-wikidata-based-knowledge-base-completion-via-llms)

### SKILL: Structured Knowledge Infusion for LLMs
Infuses structured knowledge by training T5 on factual triples:
- Models pre-trained on Wikidata outperform baselines on FreebaseQA and WikiHop
- Source: [arXiv 2205.08184](https://ar5iv.labs.arxiv.org/html/2205.08184)

---

## 16. Multi-Agent Protocols (2025)

### Emerging Standards

| Protocol | Developer | Purpose |
|----------|-----------|---------|
| **A2A** (Agent-to-Agent) | Google | Open standard for agent interaction; defines AgentCards, async task messaging, artifact delivery |
| **MCP** (Model Context Protocol) | Anthropic | Structured I/O with session context and tool schemas |
| **ACP** (Agent Communication Protocol) | Various | Agent-to-agent collaboration |

These protocols enable heterogeneous agent ecosystems critical for ontology-driven multi-agent systems.
- Source: [arXiv - AI Agents vs Agentic AI](https://arxiv.org/abs/2505.10468)

---

## 17. Anthropic's Contextual Retrieval

### Key Innovation
Addresses failure of traditional RAG solutions that remove context when encoding:
- **Contextual Embeddings**: Adds document context to chunk embeddings
- **Contextual BM25**: Enhances keyword search with context
- Reduces failed retrievals by **49%** (or **67%** with reranking)
- Source: [Anthropic - Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

### Memory Tool for Long-Context Management
File-based system enabling persistent knowledge across conversations:
- Memory tool + context editing: **39% improvement** over baseline
- **84% token consumption reduction** in 100-turn evaluations
- Source: [Anthropic - Context Management](https://anthropic.com/news/context-management)

---

## 18. Entity Linking: The Critical Bridge

Entity linking is described as "the unique and critical capability needed to combine knowledge graphs and LLMs"—the task of associating entity mentions in text to concrete entity identifiers in graphs.

### Key Tools
- **Relik**: Library for entity linking and relationship extraction with Wikipedia as target KB
- **LLM + KG Hybrid Pipelines**: Chain-of-thought prompting with factual KG retrieval achieves better entity disambiguation

### Challenge
LLMs can recognize relationships but cannot perform entity linking well on their own. Without it, impossible to use graph data to interpret new events or extend graphs.
- Source: [Metaphacts Blog](https://blog.metaphacts.com/using-knowledge-graph-based-llm-for-relation-event-detection)

---

## 19. Notable Researchers (Additional)

### Denny Vrandečić
Co-founder of Wikidata. Keynote at Knowledge Graph Conference: "The Future of Knowledge Graphs in a World of Large Language Models."

### Lilian Weng
Research lead at OpenAI. Author of influential blog posts on LLM agents and hallucination, including the widely-cited "LLM Powered Autonomous Agents" post.
- Blog: [lilianweng.github.io](https://lilianweng.github.io/posts/2023-06-23-agent/)

### Kevin Meng (MIT)
Lead researcher on ROME and MEMIT knowledge editing methods.

### Jonathan Berant
Professor at Tel Aviv University. Key contributor to semantic parsing for QA, including Neural Symbolic Machines.

---

## 20. Key GitHub Repositories

| Repository | Focus |
|------------|-------|
| [Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM) | Graph-related LLM papers |
| [Awesome-KBQA](https://github.com/RUCAIBox/Awesome-KBQA) | KBQA papers and resources |
| [Causal-LLM-Paper](https://github.com/wan19990901/Causal-LLM-Paper) | KG + causal inference + LLM papers |
| [KG-MM-Survey](https://github.com/zjukg/KG-MM-Survey) | Multimodal KG survey |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | Knowledge editing framework |
| [KRLPapers](https://github.com/thunlp/KRLPapers) | Knowledge representation learning |
| [awesome-commonsense](https://github.com/yuchenlin/awesome-commonsense) | Machine commonsense reasoning |

---

## 21. Emerging Research Directions Not in Main Survey

### 1. Neuro-Symbolic AI Meta-Cognition
Only **5% of papers** explore meta-cognition in neuro-symbolic systems—identified as the least explored area. Critical for self-improving systems.
- Source: [ScienceDirect - Neuro-symbolic AI Review](https://www.sciencedirect.com/science/article/pii/S2667305325000675)

### 2. Schema-Based vs. Schema-Free Hybrid Approaches
Best results from combining:
- Schema-based: Structure-first using predefined ontologies
- Schema-free: Flexibility-first where LLMs discover structure
- Source: [arXiv - LLM-empowered KG construction survey](https://arxiv.org/abs/2510.20345)

### 3. Inductive Link Prediction
Handling unseen entities and relations in evolving KGs:
- Entity-independent modeling with triple-view graphs
- Contrastive learning for relation-context modeling
- Critical for dynamic domains (healthcare, e-commerce, social media)
- Source: [ScienceDirect - Unified link prediction modeling](https://www.sciencedirect.com/science/article/abs/pii/S0957417425009789)

### 4. EnCompass Framework (MIT CSAIL + Asari AI)
Automatically backtracks if LLMs make mistakes during reasoning; can clone program runtime to explore solutions in parallel.
- Source: [TechXplore - EnCompass](https://techxplore.com/news/2025-12-ai-agents-results-large-language.html)

---

## Summary of Key Gaps in Main Survey

1. **Knowledge editing methods** (ROME, MEMIT) not covered
2. **Commonsense knowledge bases** (ConceptNet, ATOMIC, COMET) briefly mentioned but underexplored
3. **GNN + LLM integration** not covered in depth
4. **Self-RAG and adaptive retrieval** mechanisms absent
5. **Causal reasoning** frameworks underrepresented
6. **Multimodal knowledge graphs** only briefly touched
7. **Table Augmented Generation (TAG)** not mentioned
8. **Entity linking** as critical bridge not emphasized
9. **Knowledge graph embedding foundations** (TransE, RotatE, etc.) not covered
10. **KBQA paradigms** (IR vs NSP) not detailed

---

*Research conducted December 2025*
