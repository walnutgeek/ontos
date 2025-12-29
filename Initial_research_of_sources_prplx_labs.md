<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \# Ontos

Little research project that focused on how ontology, mind maps, or knowledge graphs
and how they could be useful for better reasoning and better context engineering for
LLMs, or even creating set of collaborating LLM agents that augment their own knowledge
graph and provide reasoning based on that.

You are helpful research assistant you want to find projects, videos, podcasts and
notable personalities operating in the same idea space or that focused on the similar
approach.

Create lists. Summarize. Point out interesting and novel ideas.

Now I have sufficient information to create a comprehensive research report. Let me organize all the findings into a well-structured document.

# Ontology, Knowledge Graphs, and LLM Reasoning: A Comprehensive Research Survey

This research survey presents an extensive exploration of how ontologies, mind maps, and knowledge graphs are being leveraged to enhance reasoning capabilities and context engineering for Large Language Models (LLMs), particularly in the context of multi-agent AI systems. The field has experienced rapid growth since 2023, driven by the convergence of symbolic AI methods with neural language models—a paradigm known as neurosymbolic AI.

## Executive Summary

The integration of ontologies and knowledge graphs with LLMs represents a fundamental shift in how AI systems reason and maintain context. Unlike pure statistical approaches, this hybrid methodology combines the semantic precision of symbolic knowledge representation with the flexible language understanding of neural networks. Research consistently demonstrates that **knowledge graph-enhanced systems achieve 35-80% improvements in accuracy** compared to vector-only retrieval systems. This survey identifies key projects, researchers, methodologies, and tools advancing this space, with particular emphasis on practical implementations for reasoning engines and collaborative agent architectures.[^1_1][^1_2][^1_3][^1_4]

## The Foundational Paradigm: Neurosymbolic AI

### Core Concepts

**Neurosymbolic artificial intelligence** merges two historically distinct approaches: symbolic reasoning (logic, rules, ontologies) and neural learning (pattern recognition, deep learning). This fusion addresses critical limitations of LLMs operating in isolation—namely hallucination, inability to perform multi-hop reasoning, and lack of explainability.[^1_5][^1_6][^1_7][^1_3][^1_8][^1_9]

**Ontologies** serve as the formal semantic layer that defines concepts, relationships, and logical rules within a domain. Using standards like the **Web Ontology Language (OWL)** and **Semantic Web Rule Language (SWRL)**, ontologies enable **inference mechanisms** that can deduce new knowledge from existing facts. For instance, if an ontology specifies that "all cats are mammals" and "all mammals are animals," a reasoning engine can infer that "all cats are animals" even without explicit statement.[^1_6][^1_10][^1_1]

**Knowledge graphs** operationalize ontologies by instantiating them with actual data—representing entities as nodes and relationships as edges in a graph structure. Unlike traditional databases, knowledge graphs excel at capturing complex, interconnected relationships and enabling traversal-based reasoning.[^1_10][^1_3][^1_11][^1_12][^1_5]

### Why This Matters for LLMs

LLMs struggle with several critical tasks that knowledge graphs naturally support:[^1_7][^1_3][^1_5]

1. **Deterministic queries**: Questions requiring exhaustive enumeration (e.g., "List all account executives in Asia") are unreliable for LLMs alone but trivial for structured graph queries[^1_5]
2. **Multi-hop reasoning**: Following chains of relationships across multiple entities (e.g., "Which suppliers provide components for Project X through Vendor Y?")[^1_2][^1_12][^1_5]
3. **Temporal reasoning**: Understanding how facts change over time and maintaining historical context[^1_13][^1_14][^1_15]
4. **Disambiguation**: Distinguishing between entities with identical names (e.g., "Reddit" as a customer vs. advertising platform)[^1_5]
5. **Hallucination reduction**: Grounding responses in verified, structured knowledge rather than probabilistic pattern matching[^1_3][^1_2][^1_7][^1_5]

## Major Research Directions and Novel Ideas

### 1. GraphRAG: Structured Retrieval for Enhanced Generation

**GraphRAG** (Graph Retrieval-Augmented Generation) represents one of the most significant advances in combining knowledge graphs with LLM generation. Pioneered by Microsoft Research in 2024, GraphRAG extracts knowledge graphs from text using LLMs, then leverages graph structure for retrieval rather than semantic similarity alone.[^1_11][^1_16][^1_17][^1_18][^1_3]

**Key innovations**:

- **Community detection algorithms** (like Louvain) cluster densely connected nodes, then generate summaries for each community[^1_2][^1_11]
- **Hierarchical summarization** enables both global queries (requiring holistic understanding) and local queries (focusing on specific entities)[^1_17][^1_3][^1_11]
- **Hybrid retrieval** combines graph traversal, vector similarity, and keyword search for comprehensive context[^1_19][^1_3][^1_11]

Benchmark results show GraphRAG achieving **80% correct answers** compared to 50.83% for traditional vector RAG, with improvements up to 90% when including partially correct responses. In complex technical domains, accuracy improvements exceeded **90% correct answers versus 47% for vector-only approaches**.[^1_3]

### 2. Ontology-Guided Reasoning and Reverse Thinking

A novel approach from recent research introduces **Ontology-Guided Reverse Thinking** for knowledge graph question answering. Instead of forward chaining from entities to answers, this method:[^1_20]

1. Identifies the desired answer type (the "aim")
2. Works backwards through the ontology schema to construct **label reasoning paths**
3. Prunes irrelevant branches by checking for condition labels
4. Uses these paths to guide targeted queries in the knowledge graph
5. Aggregates results through the LLM

This "reverse-oriented approach" achieves **state-of-the-art performance** by reducing the search space and eliminating unnecessary reasoning branches early. The technique is particularly effective for multi-hop questions requiring complex relational traversal.[^1_20]

### 3. Temporal Knowledge Graphs for Agent Memory

**Zep's Graphiti framework** introduces temporally-aware knowledge graphs specifically designed for AI agent memory. Unlike static knowledge graphs, Graphiti:[^1_14][^1_21][^1_13]

- **Tracks temporal validity** of facts, marking when relationships were true and when they became invalid[^1_22][^1_13][^1_14]
- **Automatically invalidates outdated information** based on temporal logic when new conflicting facts emerge[^1_13][^1_14][^1_22]
- **Enables incremental updates** without batch recomputation, supporting real-time agent environments[^1_22][^1_13]
- **Combines hybrid retrieval** methods: semantic search, BM25 keyword matching, and graph traversal[^1_14][^1_13]

In comprehensive benchmarks, Graphiti-powered agents achieved **94.8% accuracy** on the Deep Memory Retrieval benchmark versus 93.4% for MemGPT, with **up to 18.5% improvements** on complex temporal reasoning tasks and **90% reduction in response latency**.[^1_13]

### 4. MindMap: Knowledge Graph Prompting for Transparent Reasoning

The **MindMap framework** enables LLMs to construct their own reasoning graphs from knowledge graph inputs, creating transparent "mind maps" of their inference process. This approach:[^1_23][^1_24][^1_25]

- Extracts evidence sub-graphs from knowledge graphs based on query keywords
- Prompts LLMs to aggregate and consolidate retrieved sub-graphs into unified reasoning graphs
- Uses these graphs to guide generation while maintaining traceability to source knowledge
- Explicitly reveals reasoning pathways based on the ontology structure

MindMap demonstrates that LLMs can **synergistically reason** with both implicit (learned) knowledge and explicit (knowledge graph) knowledge, with substantial performance improvements on complex multi-hop questions. The framework is particularly valuable for medical and scientific domains where reasoning transparency is critical.[^1_24][^1_25]

### 5. Multi-Agent Ontology Engineering with LLMs

**Agent-OM (Agent for Ontology Matching)** introduces a paradigm shift in ontology alignment by using autonomous LLM agents. The system consists of:[^1_26][^1_27][^1_28]

- **Siamese agents** (Retrieval Agent and Matching Agent) that work in parallel
- **Planning modules** that use Chain-of-Thought decomposition for matching tasks
- **Hybrid database storage** combining relational and vector databases for entity information
- **Tool-based matching** rather than pure conversational dialogue

This approach achieves results **very close to the best long-standing performance** in Ontology Alignment Evaluation Initiative (OAEI) benchmarks, while dramatically reducing human effort in the matching process.[^1_27][^1_28][^1_26]

### 6. Dynamic Mind Maps for Evolving Knowledge

Research from Bigdata.com demonstrates **dynamic mind maps** that evolve with real-time data using Large Reasoning Models (LRMs) grounded in live search APIs. The system:[^1_29]

- Generates baseline mind maps using fast LLMs (e.g., GPT-4o-mini)
- Refines them monthly using LRMs (GPT-o3-mini) grounded in recent news
- Automatically surfaces **emerging risks and narratives** not explicitly prompted
- Powers agentic retrieval pipelines with up-to-date thematic structure

This approach transforms mind maps from static conceptual tools into **self-updating analytical infrastructures** that maintain relevance as information landscapes shift. The grounded LRM adds branches grounded in real-world signals, ensuring the knowledge structure reflects actual developments rather than just model knowledge.[^1_29]

### 7. Cognee: Memory-First Architecture with Ontology Integration

**Cognee** represents an emerging open-source AI memory system that combines graph-based extraction with RDF/OWL ontology validation. Key features include:[^1_30][^1_31][^1_32][^1_33]

- **Automatic graph extraction** from text, with entity and relationship identification via LLMs
- **Ontology grounding**: Optional RDF/OWL files validate extracted entities against canonical vocabularies[^1_33][^1_34]
- **Hybrid pipelines**: Composable workflows linking parsing, scraping, and search[^1_31][^1_32]
- **Natural language querying**: Users can search the graph without writing Cypher queries[^1_32][^1_31]

By integrating ontologies, Cognee achieves **enhanced semantic understanding** through class inheritance, transitive reasoning, and domain enrichment. The system reports accuracy improvements from **60% (RAG-only) to ~90%** with the memory-first graph architecture.[^1_34][^1_31]

## Notable Researchers and Thought Leaders

### Pascal Hitzler

Professor at Kansas State University, Hitzler is a leading figure in modular ontology engineering and LLM-assisted knowledge graph construction. His work on **Modular Ontology Modeling (MOMo)** emphasizes pattern-based approaches that make ontology engineering accessible to domain experts rather than requiring deep OWL expertise.[^1_35][^1_36][^1_37][^1_38]

### Cogan Shimizu

Assistant Professor at Wright State University, Shimizu developed **CoModIDE** (Comprehensive Modular Ontology IDE), a Protégé plugin enabling graphical drag-and-drop ontology design. His research explores how modular structures allow LLMs to reason about ontologies more effectively by breaking them into human-comprehensible chunks.[^1_36][^1_38][^1_39][^1_40][^1_41][^1_35]

### Juan Sequeda

Principal Scientist at data.world, Sequeda has demonstrated that **knowledge graphs increase LLM accuracy by 3-4 times** in enterprise question answering over relational databases. His benchmark work on insurance domain schemas showed accuracy improvements from **17% (LLM alone) to 72.6% with ontology-based query validation**. Sequeda advocates for treating semantics as a "first-class citizen" in enterprise data architectures.[^1_42][^1_4][^1_43][^1_44]

### Paul Groth

Professor at University of Amsterdam, Groth's research spans knowledge graphs, data provenance, and the impact of LLMs on data engineering. He explores how LLMs can simplify knowledge graph construction while maintaining quality and verifiability.[^1_45][^1_46]

### Anna Fensel

Associate Professor at Wageningen University and University of Innsbruck, Fensel's work focuses on semantic technologies for life sciences, health care, and agriculture. She emphasizes knowledge graphs as tools for FAIR principles (Findable, Accessible, Interoperable, Reusable) and legal compliance in data-intensive domains.[^1_47][^1_48][^1_49][^1_50]

### Helena Deus

Deus published influential work on **neurosymbolic AI for systems with bounded rationality**. Her framework positions knowledge graphs as computational representations of system topology, enabling AI agent orchestrators to predict system resilience and adjust agent behaviors without causing systemic collapse. This perspective is particularly relevant for enterprise environments with complex interdependencies.[^1_8][^1_51][^1_52]

## Key Projects and Tools

### Open Source Frameworks

**1. Graphiti (Zep)**

- **Purpose**: Temporal knowledge graph framework for AI agent memory
- **Language**: Python
- **Key Features**: Real-time incremental updates, hybrid retrieval, temporal invalidation
- **Performance**: 94.8% accuracy on DMR benchmark, 90% latency reduction[^1_14][^1_22][^1_13]
- **Repository**: github.com/getzep/graphiti[^1_53]

**2. Cognee**

- **Purpose**: Memory-augmented LLM architecture with RDF/OWL ontology support
- **Language**: Python
- **Key Features**: Custom ontologies, composable pipelines, embeddings, entity extraction
- **Accuracy**: ~90% vs. 60% for traditional RAG[^1_31]
- **Website**: cognee.ai[^1_54]
- **Documentation**: docs.cognee.ai[^1_33]

**3. MindMap (by Wyl-willing)**

- **Purpose**: Knowledge graph prompting for LLM graph-of-thoughts reasoning
- **Language**: Python
- **Key Features**: Evidence sub-graph mining, graph aggregation, transparent reasoning paths
- **Repository**: github.com/wyl-willing/MindMap[^1_25][^1_23][^1_24]

**4. CoModIDE**

- **Purpose**: Graphical ontology design pattern editor for Protégé
- **Key Features**: Drag-and-drop pattern instantiation, OPLa annotations, schema diagrams
- **Methodology**: Modular Ontology Modeling (MOMo)
- **Website**: comodide.com[^1_40][^1_41][^1_55][^1_56]

**5. FalkorDB with GraphRAG-SDK**

- **Purpose**: Graph database with automated ontology management
- **Key Features**: Property graph model, automatic ontology detection from knowledge graphs, LLM integration
- **SDK Version**: 0.5+ with automatic ontology loading[^1_57][^1_58][^1_59][^1_60]

**6. EscherGraph (PinkDot.ai)**

- **Purpose**: Knowledge graph construction from documents using ontologies
- **Approach**: Ontology-first design to define extraction scope, avoiding context loss from chunking
- **Repository**: github.com/PinkDot-AI/eschergraph[^1_61][^1_62]


### Commercial Platforms

**Stardog**

- **Platform**: Enterprise Knowledge Graph with LLM Voicebox
- **Voicebox Capabilities**:
    - Natural language question answering over knowledge graphs
    - Automated ontology creation from plain language prompts
    - Information extraction for knowledge graph completion
    - Post-generation hallucination detection
- **Architecture**: Fusion of Knowledge Graph + LLM for six core GenAI jobs (querying, grounding, guiding, constructing, completing, guarding)[^1_63][^1_64][^1_65][^1_66][^1_67]

**Neo4j**

- **Graph Database**: Property graph model with limited neosemantic tools for ontology storage
- **GraphRAG Support**: Official GraphRAG manifesto cites knowledge graph accuracy research[^1_4][^1_3]
- **Agent Integration**: Used extensively in LangChain/LangGraph workflows[^1_12][^1_68]

**Amazon Bedrock with Neptune**

- **Service**: Managed GraphRAG support integrated with Amazon Neptune Analytics
- **Availability**: Built into Amazon Bedrock Knowledge Bases (preview as of Dec 2024)[^1_3]


## Frameworks and Libraries

### LangGraph (LangChain)

**Purpose**: Agent orchestration framework with graph-based stateful workflows[^1_69][^1_70][^1_71][^1_72][^1_73]

**Core Concepts**:

- **Nodes**: Discrete computation units (functions, agents, tools)
- **Edges**: Control flow (direct or conditional)
- **State**: Shared data/context across all nodes
- **Cycles**: Iterative agent-tool communication loops

**Knowledge Graph Integration**: LangGraph agents can treat knowledge graphs as tools, querying them during reasoning loops. The framework supports human-in-the-loop interventions and persistence, making it suitable for enterprise agentic applications involving knowledge graphs.[^1_70][^1_69]

### LangChain

**Capabilities**: Chains, agents, and tools for LLM applications
**Knowledge Graph Support**: Native integrations for Neo4j, graph chain modules, and question-answering over graphs[^1_74][^1_75]

**Limitations vs. LangGraph**: Traditional LangChain chains are DAG-based and less suitable for complex multi-step reasoning with cycles. LangGraph addresses this by supporting cyclic workflows essential for agent-knowledge graph interactions.[^1_69]

## Conferences, Workshops, and Community Events

### Text2KG Workshop Series

- **Full Name**: LLM-Integrated Knowledge Graph Generation from Text (Text2KG)
- **Affiliation**: Extended Semantic Web Conference (ESWC)
- **Recent Edition**: 4th workshop, June 1-5, 2025, Portorož, Slovenia
- **Focus**: Entity recognition, relation extraction, LLM-driven inference, hallucination mitigation, industrial applications
- **Proceedings**: Available at ceur-ws.org (2022, 2023, 2024 editions)[^1_76]


### LLMs and Ontologies Workshop (ONTOLLM)

- **Full Name**: 2nd Annual Workshop on LLMs and Ontologies
- **Affiliation**: Joint Ontologies Workshop (JOWO) / FOIS 2025
- **Date**: September 2025
- **Focus**: Convergence of knowledge representation and LLM strategies, design patterns, benchmarks
- **Website**: cbp1012.github.io/JOWO-ONTOLLM/[^1_77]


### Knowledge Graph Conference (KGC)

- **Dates**: May 4-8, 2026
- **Location**: Cornell Tech NYC + Online
- **Scale**: 2,000+ attendees
- **Highlights**: Hands-on workshops (most popular feature), Healthcare \& Life Sciences symposium
- **Focus**: Enterprise AI, LLMs, NLP, machine learning, data management
- **Website**: knowledgegraph.tech[^1_78]


### International Joint Conference on Knowledge Graphs (IJCKG)

- **Edition**: 14th conference (IJCKG 2025)
- **Focus**: KG4LLM and LLM4KG, knowledge representation, graph databases, machine learning on graphs
- **Topics**: Knowledge graph enhanced LLMs, ontology modeling, semantic search, question answering
- **Website**: ijckg2025.github.io[^1_79]


### Knowledge Graphs and Big Data Workshop

- **Edition**: 5th workshop (Virtual, December 2025)
- **Topics**: Graph retrieval augmented generation, ontology population, link prediction, code-driven ontology systems
- **Examples**: GRAFT framework, Code2Onto multi-agent system, HyperComplEx embeddings[^1_80]


### SEMANTiCS Conference

- **Historical Note**: Top-tier European conference for semantic technologies
- **Chairs**: Anna Fensel has served in chair roles at SEMANTiCS and ESWC[^1_50]


## Podcasts and Video Resources

### Podcasts

**1. Knowledge Graph Insights**

- **Platform**: Apple Podcasts, Spotify
- **Host**: Larry (surname not specified in sources)
- **Format**: Expert interviews on semantic technology, ontology engineering, linked data
- **Notable Episodes**:
    - Episode 19: Juan Sequeda on LLMs as critical enabler for KG adoption[^1_4]
    - Episode 37: Chris Mungall on collaborative KGs in life sciences[^1_81]
    - Episode 38: Casey Hart on philosophical foundations of ontology practice[^1_81]
- **Website**: knowledgegraphinsights.com[^1_81]

**2. Catalog and Cocktails**

- **Co-host**: Juan Sequeda
- **Description**: "Honest, no-bs, non-salesy data podcast"[^1_4]

**3. AI Engineering Podcast**

- **Episode**: "Enhancing AI Retrieval with Knowledge Graphs: A Deep Dive into GraphRAG"
- **Guest**: Philip Rathle, CTO of Neo4J
- **Topics**: GraphRAG, ontology use cases, entity extraction, infrastructure considerations[^1_74]

**4. Data Engineering Podcast**

- **Episode**: "From Academia to Industry: Bridging Data Engineering Challenges"
- **Guest**: Professor Paul Groth (University of Amsterdam)
- **Topics**: Knowledge graphs, data provenance, LLM impact on data engineering[^1_45]

**5. D3Clarity: Ontology vs. Taxonomy Podcast**

- **Hosts**: Data Dave, Alexis, Erik Lee
- **Date**: March 18, 2025
- **Duration**: 22:55
- **Topics**: Ontology, taxonomy, knowledge graphs for business insights, AI applications[^1_82]

**6. Chaos Orchestra - The Knowledge Graph Podcast**

- **Platform**: Spotify
- **Topics**: Graph representation models, ontology engineering, enterprise-wide knowledge graphs[^1_83]


### YouTube Channels and Videos

**1. Microsoft Reactor: Intro to GraphRAG**

- **Date**: September 7, 2024
- **Duration**: 1:01:05
- **Content**: Understanding GraphRAG, environment setup, LangChain implementation, Neo4j integration
- **Presenters**: John (surname not specified) and Apera from Microsoft[^1_17]

**2. Microsoft Research: GraphRAG Research Paper Reading**

- **Presenter**: Ganges (from SupportVectors channel)
- **Duration**: ~40 minutes
- **Content**: Code walkthrough, LLM-generated graphs, reasoning capabilities, community traversal algorithms[^1_84]

**3. Glean: Working AI - Knowledge Graph**

- **Date**: August 19, 2025
- **Duration**: 13:31
- **Content**: How knowledge graphs enable complex work for AI agents, capturing relationships LLMs miss[^1_19]

**4. CocoIndex: Build Real-Time Knowledge Graph For Documents with LLM**

- **Date**: May 12, 2025
- **Duration**: 22:21
- **Content**: Hands-on coding walkthrough, entity extraction, relationship mapping, Neo4j export
- **Repository**: github.com/cocoindex-io/cocoindex[^1_85]

**5. Stardog: LLMs + Knowledge Graphs - Enabling Trustworthy AI Agents**

- **Date**: August 15, 2025
- **Duration**: 37:00
- **Presenter**: Mike Grove (SVP Engineering \& Co-Founder, Stardog)
- **Content**: Semantic layers, data fabrics, ontology-backed reasoning, live demo with Wizbox[^1_86]

**6. Dr. Cogan Shimizu: Accelerating Knowledge Graph and Ontology Engineering with LLMs**

- **Date**: April 9, 2025
- **Content**: Modular ontologies, pattern-based approaches, LLM-powered ontology alignment
- **Institution**: CASTLE Lab (Knowledge Architecture, Structures, Techniques, Learning, and Evaluation Laboratory)[^1_36]

**7. Neo4j: Going Meta S02E05 – One Ontology to Rule Them All**

- **Date**: January 7, 2025
- **Guest**: Jesús Barrasa
- **Topic**: Building knowledge graphs from mixed data using unified ontology approaches[^1_87]

**8. AI Engineer: Practical GraphRAG - Making LLMs smarter with Knowledge Graphs**

- **Date**: July 22, 2025
- **Duration**: 19:46
- **Content**: GraphRAG challenges, implementation patterns, agentic examples with Google's ADK[^1_88]

**9. Knowledge Graphs, Kuzu, and Building Smarter Agents (S2E4)**

- **Date**: September 23, 2025
- **Guests**: Michael and Prashanth
- **Topics**: Kuzu's embedded columnar architecture, LLMs for graph construction, Cypher translation[^1_89]

**10. Anna Fensel: Knowledge graphs for FAIR principles**

- **Date**: March 6, 2025
- **Institution**: Wageningen University
- **Topics**: Semantic web in healthcare/life sciences, knowledge graphs for legal compliance, FAIR data principles[^1_49]


## Benchmark and Evaluation Resources

### DMR (Deep Memory Retrieval) Benchmark

- **Origin**: Established by MemGPT team
- **Purpose**: Evaluate agent memory systems
- **Best Performance**: Zep (Graphiti) at 94.8% vs. MemGPT at 93.4%[^1_13]


### LongMemEval Benchmark

- **Focus**: Enterprise use cases with complex temporal reasoning
- **Tasks**: Cross-session information synthesis, long-term context maintenance
- **Results**: Zep achieved up to 18.5% accuracy improvements with 90% latency reduction[^1_13]


### OAEI (Ontology Alignment Evaluation Initiative)

- **Purpose**: Annual competition for ontology matching systems
- **Agent-OM Performance**: Results "very close to the best long-standing performance"[^1_28][^1_26][^1_27]


### Juan Sequeda's Enterprise QA Benchmark

- **Domain**: Insurance enterprise schema
- **Question Types**: Fact-based, multi-hop, numerical, tabular, temporal, multi-constraint
- **Scale**: 43 enterprise-grade questions
- **Results**:
    - LLM alone: 17% execution accuracy
    - LLM + Knowledge Graph: 72.6% execution accuracy
    - **4.2x improvement** with ontology-based query validation[^1_42][^1_4]


### Lettria GraphRAG Benchmarks

- **Domains**: Finance (Amazon reports), healthcare (COVID-19 studies), aerospace (technical specs), law (EU directives)
- **Results**:
    - Traditional RAG: 50.83% correct answers
    - GraphRAG: 80% correct answers
    - With acceptable answers: 67.5% (RAG) vs. ~90% (GraphRAG)
    - Industry sector: 90.63% (GraphRAG) vs. 46.88% (vector RAG)[^1_3]


## Industry Applications and Use Cases

### Financial Services

- **Risk Analysis**: Dynamic mind maps track emerging risks grounded in real-time news[^1_29]
- **Thematic Screening**: Agent-powered portfolio exposure analysis using evolving knowledge graphs[^1_29]
- **Audit Research**: GraphRAG as research assistant over financial regulations and auditing standards[^1_90]


### Healthcare and Life Sciences

- **Biomedical QA**: KGARevion agent for knowledge-intensive questions using graph-based triplet generation[^1_91]
- **Symptom Ontology Extension**: Multi-agent LLM framework for online clustering and taxonomy updates[^1_92]
- **Drug Discovery**: Neurosymbolic AI combining deep neural networks with ontological constraints for molecular design[^1_8]


### Enterprise Data Management

- **Data Integration**: Knowledge graphs as semantic layer connecting heterogeneous data sources[^1_2][^1_42][^1_5]
- **Metadata Management**: Unified catalog with ontology-backed relationships[^1_12][^1_4]
- **Compliance and Governance**: Knowledge graphs for GDPR consent tracking and data contracts[^1_48]


### Sales and CRM

- **Lead Management**: Knowledge graphs storing prospect preferences, interactions, buying signals[^1_14]
- **Context-Aware Outreach**: Agents understanding deal history and engagement patterns[^1_14]


## Novel Theoretical Contributions

### 1. Bounded Rationality and System Topology

Helena Deus's framework positions **bounded rationality** as the key driver for neurosymbolic AI in complex systems. Each agent in an enterprise has incomplete information about distant parts of the system. Knowledge graphs represent the system's **topology**—the network of agents and relationship strengths—enabling orchestrators to:[^1_8]

- Predict system resilience under different interventions
- Adjust agent goals to prevent systemic collapse
- Incorporate new rules immediately through symbolic updates (avoiding latency of neural retraining)

This perspective shifts knowledge graphs from static reference tools to **dynamic control structures** for multi-agent coordination.[^1_8]

### 2. Ontology as Bridge Between Human and Machine Conceptualization

Research emphasizes ontologies as **modules that correspond to human knowledge partitions**. This "bridge" metaphor is central to modular ontology modeling:[^1_38][^1_36]

- Modules are self-contained conceptual units aligned with how humans think about topics
- LLMs can reason more effectively when ontologies are chunked into human-scale pieces
- Schema diagrams provide intuitive visual representations that guide both human designers and automated systems[^1_38][^1_40][^1_36]


### 3. Temporal Knowledge as First-Class Citizen

Traditional knowledge graphs treat facts as eternal truths. The **temporal knowledge graph paradigm** introduced by Graphiti treats temporal validity as a first-class property:[^1_13][^1_14]

- Facts have validity periods (when they were true)
- New information can invalidate old facts without deletion (preserving history)
- Agents can reason about change over time ("What was the user's preference last month vs. now?")

This temporal dimension is critical for **long-term agent memory** in dynamic environments where facts evolve.[^1_14][^1_13]

### 4. Schema-Based vs. Schema-Free Knowledge Construction

Recent LLM-empowered KG construction literature identifies two complementary paradigms:[^1_93]

- **Schema-based**: Structure-first approaches using predefined ontologies for consistency and precision
- **Schema-free**: Flexibility-first approaches where LLMs discover structure through reasoning and clustering

Hybrid approaches that combine both—using schema guidance where available and open discovery where needed—show the most promise for enterprise applications.[^1_93]

## Implementation Patterns and Best Practices

### Pattern 1: Graph RAG Pipeline

1. **Extraction**: Use LLMs to identify entities and relationships from text
2. **Graph Construction**: Store as triplets (subject-predicate-object) in graph database
3. **Community Detection**: Apply algorithms (Louvain) to cluster related entities
4. **Summarization**: Generate natural language summaries for communities
5. **Hybrid Retrieval**: Combine graph traversal + vector similarity + keyword search
6. **LLM Generation**: Provide retrieved context to LLM for final response[^1_11][^1_12][^1_2][^1_3]

### Pattern 2: Ontology-First Knowledge Graph Construction

1. **Define Ontology**: Specify entity types and relationships relevant to domain
2. **LLM Extraction**: Use ontology as schema for targeted entity/relationship extraction
3. **Validation**: Check extracted triplets against ontology rules
4. **Population**: Instantiate ontology with actual data
5. **Reasoning**: Apply inference rules to derive implicit knowledge[^1_58][^1_92][^1_61][^1_34]

### Pattern 3: Temporal Graph Updates

1. **Ingest New Information**: Chat messages, JSON data, documents
2. **Extract Entities/Relations**: LLM identifies new facts
3. **Graph Matching**: Find existing similar nodes to avoid duplication
4. **Temporal Extraction**: Identify timestamps or date references
5. **Edge Insertion**: Add new facts with temporal metadata
6. **Invalidation**: Mark conflicting old edges as invalid based on temporal logic[^1_21][^1_13][^1_14]

### Pattern 4: Modular Ontology Engineering with Patterns

1. **Identify Key Notions**: Break domain into conceptual components
2. **Match Patterns**: Find relevant ODPs (Ontology Design Patterns) from libraries
3. **Instantiate Modules**: Create modules from patterns, adapting as needed
4. **Systematic Axiomatization**: Apply 17 common axioms to node-edge-node constructs
5. **Assembly**: Connect modules with inter-module axioms
6. **Refinement**: Improve entity names, check axiom appropriateness[^1_41][^1_40][^1_36][^1_38]

## Tools and Infrastructure Ecosystem

### Graph Databases

- **Neo4j**: Most popular property graph database, extensive ecosystem
- **Amazon Neptune**: Managed graph database (RDF + property graphs)
- **FalkorDB**: Open-source with built-in GraphRAG-SDK and ontology support
- **Memgraph**: High-performance in-memory graph database
- **AllegroGraph**: RDF triplestore with neuro-symbolic AI features[^1_9]


### Ontology Editors

- **Protégé**: Leading open-source ontology editor (Stanford)
- **CoModIDE Plugin**: Graphical modular ontology engineering for Protégé[^1_55][^1_40][^1_41]
- **TopBraid Composer**: Commercial semantic modeling tool


### Vector Databases (for Hybrid Systems)

- **Pinecone**: Managed vector database with similarity search
- **Weaviate**: Open-source vector database with knowledge graph features
- **ChromaDB**: Lightweight embedding database for RAG
- **Qdrant**: High-performance vector search engine[^1_94]


### Agent Frameworks

- **LangGraph**: Stateful graph-based agent orchestration (LangChain)[^1_72][^1_69]
- **LangChain**: Chains, agents, tools for LLM applications[^1_75][^1_74]
- **AutoGen**: Microsoft's multi-agent conversation framework[^1_94]
- **CrewAI**: Role-based multi-agent collaboration[^1_94]


### Reasoning Engines

- **Pellet**: OWL DL reasoner
- **HermiT**: OWL reasoner with tableau algorithm
- **Fact++**: Description logic reasoner[^1_1][^1_6]


## Future Directions and Open Challenges

### 1. Self-Improving Knowledge Graphs

**Challenge**: Can enhanced reasoning capabilities feed back into more robust automated KG construction, creating a virtuous cycle?[^1_93]

**Research Direction**: Systems where agents use reasoning to validate and extend their own knowledge graphs, with minimal human oversight.[^1_92][^1_63]

### 2. New Mathematical Frameworks for Temporal System Evolution

**Challenge**: Linear algebra powering neural networks and graph topology may be insufficient to model dynamic systems where the topology itself evolves.[^1_8]

**Vision**: Analogous to how calculus enabled physics to model moving systems (beyond Newton's static mechanics), new mathematics may be needed for neurosymbolic AI to truly model system evolution and interdependencies.[^1_8]

### 3. Scalability of Reasoning

**Challenge**: Ontological reasoning can be computationally expensive at scale. How to balance expressiveness with performance?[^1_1][^1_2]

**Approaches**:

- Modular ontologies that isolate reasoning to relevant subgraphs[^1_36][^1_38]
- Hybrid systems that apply full reasoning only when necessary
- Approximation algorithms for large-scale inference


### 4. Ontology Discovery and Maintenance

**Challenge**: Building ontologies remains labor-intensive despite LLM assistance.[^1_95][^1_35]

**Promising Directions**:

- Fully autonomous ontology generation from unstructured data (e.g., GraphRAG to OntoRAG pipeline)[^1_93]
- Continuous ontology evolution driven by usage patterns
- Crowd-sourced ontology refinement with AI-assisted quality control[^1_63]


### 5. Interoperability Across Domains

**Challenge**: Linking knowledge graphs across different domains and organizational boundaries.[^1_96][^1_10]

**Requirements**:

- Standardized ontology alignment protocols
- Cross-domain reasoning frameworks
- Privacy-preserving knowledge sharing mechanisms


## Conclusion

The convergence of ontologies, knowledge graphs, and LLMs represents a paradigm shift from purely neural AI toward hybrid neurosymbolic systems that combine learning with reasoning. Research consistently demonstrates **substantial accuracy improvements** (35-80%) when LLMs are grounded in structured knowledge graphs rather than operating on statistical patterns alone.[^1_6][^1_9][^1_1][^1_2][^1_4][^1_3][^1_8]

Key insights from this survey include:

1. **Ontologies provide the semantic layer** that enables inference, validation, and explainability—capabilities LLMs alone cannot reliably achieve[^1_10][^1_6][^1_1][^1_5]
2. **Knowledge graphs operationalize ontologies** with actual data, enabling multi-hop reasoning, disambiguation, and deterministic queries[^1_7][^1_5][^1_3]
3. **Temporal knowledge graphs** extend this paradigm to dynamic environments where facts evolve, critical for long-term agent memory[^1_13][^1_14]
4. **Modular approaches** make ontology engineering more accessible by aligning with human conceptualization and enabling LLM-assisted development[^1_35][^1_38][^1_36]
5. **GraphRAG architectures** that combine graph structure with vector similarity achieve state-of-the-art performance on complex reasoning tasks[^1_11][^1_17][^1_3]

The ecosystem is maturing rapidly, with robust open-source tools (Graphiti, Cognee, MindMap, CoModIDE), enterprise platforms (Stardog, Neo4j), and standardized benchmarks emerging. Active research communities gather at conferences like KGC, ESWC, and IJCKG, while workshops like Text2KG and ONTOLLM foster cross-pollination between semantic web and LLM research.[^1_64][^1_30][^1_76][^1_77][^1_78][^1_53][^1_79][^1_40][^1_31][^1_14]

For practitioners building reasoning-capable LLM systems, the message is clear: **invest in semantics and knowledge graphs as foundational infrastructure**. The pendulum is swinging back toward hybrid architectures that leverage both the pattern recognition of neural networks and the logical precision of symbolic AI—a synthesis that promises more trustworthy, explainable, and capable AI systems.[^1_9][^1_6][^1_1][^1_4][^1_8]

***

## References

Ontology vs Graph Database. LLM Agents as Reasoners - Substack[^1_1]
Advancing engineering research through context-aware and knowledge graph-driven RAG - NIH[^1_2]
Ontology extension by online clustering with LLM agents - PMC[^1_92]
Ontology-Guided Reverse Thinking Makes LLMs Better - ACL Anthology[^1_20]
How knowledge graphs work and why they are the key to context for AI - Glean[^1_5]
Ontology vs Graph Database. LLM Agents as Reasoners - AI Plain English[^1_6]
Understanding Ontologies: Blueprints for Knowledge Graph Structures - FalkorDB[^1_10]
What is Context Engineering? Knowledge Graphs for LLMs - LinkedIn[^1_7]
Agents Need Democratized Data Too - Stardog[^1_63]
Turning Knowledge Graphs into Memory with Ontologies - Reddit[^1_30]
Improving Retrieval Augmented Generation accuracy with GraphRAG - AWS[^1_3]
Neurosymbolic AI: Knowledge Graphs for systems with bounded rationality - LinkedIn (Helena Deus)[^1_8]
Welcome - Microsoft GraphRAG[^1_11]
Neuro-Symbolic AI with AllegroGraph[^1_9]
Dynamic mind maps using grounded Large Reasoning Models - Bigdata.com[^1_29]
MindMap GitHub Repository[^1_23]
Accelerating Knowledge Graph and Ontology Engineering with LLMs - PDF[^1_35]
The Role of Knowledge Graphs in Building Agentic AI Systems - ZBrain[^1_12]
LLM-Integrated Knowledge Graph Generation Workshop (Text2KG 2025)[^1_76]
Unlocking Trustworthy AI: Ontologies and Knowledge Graphs - Cyberhill Partners[^1_42]
Deep Dive into Knowledge Graph Reasoning Techniques - Sparkco[^1_94]
2nd Annual Workshop on LLMs and Ontologies[^1_77]
Knowledge Graph from ontology and documents with LLMs - Reddit[^1_61]
5th Workshop on Knowledge Graphs and Big Data[^1_80]
Knowledge Graph Conference 2026[^1_78]
getzep/graphiti GitHub Repository[^1_53]
IJCKG 2025[^1_79]
MindMap: Knowledge Graph Prompting - ACL Anthology PDF[^1_24]
Agent-OM: Leveraging LLM Agents for Ontology Matching - arXiv[^1_27]
From RAG to Graphs: How Cognee is Building Self-Improving AI Memory - Memgraph[^1_31]
MindMap: Knowledge Graph Prompting - arXiv[^1_25]
Leveraging LLM Agents for Ontology Matching - VLDB PDF[^1_28]
Apache Iceberg Lakehouse with Cognee - AI Memory Guide[^1_32]
Ontologies - Cognee Documentation[^1_33]
Enhancing Knowledge Graphs with Ontology Integration - Cognee[^1_34]
Cognee - AI memory engine[^1_54]
Juan Sequeda: LLM critical enabler in knowledge graph adoption - Knowledge Graph Insights[^1_4]
Dr. Cogan Shimizu: Accelerating KG and Ontology Engineering with LLMs - YouTube[^1_36]
Accelerating Knowledge Graph and Ontology Engineering with LLMs - arXiv[^1_37]
Juan Sequeda - The Knowledge Graph Conference[^1_43]
Modular Ontology Engineering - Stanford PDF[^1_38]
Accelerating knowledge graph and ontology engineering with LLMs - ScienceDirect[^1_95]
Juan F. Sequeda Personal Website[^1_44]
Intro to GraphRAG - Microsoft Reactor YouTube[^1_17]
Ontology vs. Taxonomy: How AI Uses Knowledge Graphs - D3Clarity Podcast[^1_82]
Working AI: Knowledge Graph - Glean YouTube[^1_19]
Knowledge Graph Insights Podcast - Apple Podcasts[^1_81]
Build Real-Time Knowledge Graph For Documents with LLM - CocoIndex YouTube[^1_85]
GraphRAG: Research Paper reading - YouTube[^1_84]
LLMs + Knowledge Graphs: Enabling Trustworthy AI Agents - Stardog YouTube[^1_86]
Your First GraphRAG Demo - Microsoft Tech Community[^1_90]
Enhancing AI Retrieval with Knowledge Graphs - AI Engineering Podcast[^1_74]
Knowledge Graphs, Kuzu, and Building Smarter Agents - YouTube[^1_89]
Project GraphRAG - Microsoft Research[^1_18]
Graphiti - Temporal Knowledge Graphs for AI Agents - Zep YouTube[^1_97]
How to Use Knowledge Graph Tools to Enhance AI Development - FalkorDB[^1_57]
Zep: A Temporal Knowledge Graph Architecture for Agent Memory - arXiv[^1_13]
Zep: Context Engineering \& Agent Memory Platform[^1_14]
Using FalkorGraphRagCapability with agents - AG2 Documentation[^1_58]
How Zep Works: A Visual Guide to Knowledge Graphs for AI Agents[^1_15]
Zep - open-source Graph Memory for AI Apps - Reddit[^1_21]
GraphRAG-SDK 0.5: Simplify Knowledge Graph Integration - FalkorDB[^1_60]
Graphiti Open Source - Zep[^1_22]
From Academia to Industry: Paul Groth - Data Engineering Podcast[^1_45]
Anna Fensel - Google Scholar[^1_47]
LLM-empowered knowledge graph construction: A survey - arXiv[^1_93]
Anna Fensel - The Knowledge Graph Conference[^1_48]
Anna Fensel - Knowledge graphs for FAIR principles - YouTube[^1_49]
Prof. Dr. Anna Fensel - Google Sites[^1_50]
Helena Deus LinkedIn Post on Neurosymbolic AI[^1_51]
Machine Learning with Knowledge Graphs - Stardog[^1_64]
LangGraph Tutorial: Building LLM Agents - Zep[^1_69]
CoModIDE - The Comprehensive Modular Ontology Engineering IDE - PDF[^1_40]
Enterprise AI Requires the Fusion of LLM and Knowledge Graph - Stardog[^1_65]
Modular Ontology Modeling - Semantic Web Journal PDF[^1_41]
Stardog Voicebox FAQ[^1_66]
Welcome to CoModIDE[^1_55]
LangGraph - LangChain[^1_72]
Modular Ontology Engineering with CoModIDE Tutorial[^1_56]
<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_111][^1_112][^1_113][^1_114][^1_115][^1_116][^1_117][^1_118][^1_119][^1_120][^1_121][^1_122][^1_123][^1_124][^1_125][^1_126][^1_127][^1_128][^1_129][^1_130][^1_131][^1_132][^1_133][^1_134][^1_135][^1_136][^1_137][^1_138][^1_139][^1_140][^1_141][^1_142][^1_143][^1_144][^1_145][^1_146][^1_147][^1_148][^1_149][^1_150][^1_151][^1_152][^1_153][^1_154][^1_155][^1_156][^1_157][^1_158][^1_159][^1_160][^1_161][^1_162][^1_163][^1_164][^1_165][^1_166][^1_167][^1_168][^1_169][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: https://substack.com/home/post/p-157686573

[^1_2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12672433/

[^1_3]: https://aws.amazon.com/blogs/machine-learning/improving-retrieval-augmented-generation-accuracy-with-graphrag/

[^1_4]: http://knowledgegraphinsights.com/juan-sequeda/

[^1_5]: https://www.glean.com/blog/knowledge-graph-agentic-engine

[^1_6]: https://ai.plainenglish.io/ontology-vs-graph-database-llm-agents-as-reasoners-62bfb6008ac8

[^1_7]: https://www.linkedin.com/posts/connecteddataworld_knowledgegraphs-graphrag-semanticai-activity-7343257041609588737-XArI

[^1_8]: https://www.linkedin.com/pulse/neurosymbolic-ai-knowledge-graphs-systems-bounded-helena-deus-phd-rhove

[^1_9]: https://allegrograph.com/products/neuro-symbolic-ai/

[^1_10]: https://www.falkordb.com/blog/understanding-ontologies-knowledge-graph-schemas/

[^1_11]: https://microsoft.github.io/graphrag/

[^1_12]: https://zbrain.ai/knowledge-graphs-for-agentic-ai/

[^1_13]: https://arxiv.org/abs/2501.13956

[^1_14]: https://www.getzep.com

[^1_15]: https://blog.getzep.com/a-visual-guide-to-knowledge-graphs-for-ai-agents/

[^1_16]: https://arxiv.org/abs/2501.00309

[^1_17]: https://www.youtube.com/watch?v=f6pUqDeMiG0

[^1_18]: https://www.microsoft.com/en-us/research/project/graphrag/

[^1_19]: https://www.youtube.com/watch?v=MsZUIi97ynk

[^1_20]: https://aclanthology.org/2025.acl-long.741.pdf

[^1_21]: https://www.reddit.com/r/LLMDevs/comments/1fq302p/zep_opensource_graph_memory_for_ai_apps/

[^1_22]: https://www.getzep.com/product/open-source/

[^1_23]: https://github.com/wyl-willing/MindMap

[^1_24]: https://aclanthology.org/2024.acl-long.558.pdf

[^1_25]: https://arxiv.org/abs/2308.09729

[^1_26]: https://arxiv.org/html/2312.00326v13

[^1_27]: https://arxiv.org/abs/2312.00326

[^1_28]: https://www.vldb.org/pvldb/vol18/p516-qiang.pdf

[^1_29]: https://bigdata.com/blog/dynamic-mind-maps-using-grounded-large-reasoning-models

[^1_30]: https://www.reddit.com/r/MachineLearning/comments/1jot2zr/dp_turning_knowledge_graphs_into_memory_with/

[^1_31]: https://memgraph.com/blog/from-rag-to-graphs-cognee-ai-memory

[^1_32]: https://www.cognee.ai/blog/deep-dives/iceberg-lakehouse-with-cognee-tower-ai-memory-guide

[^1_33]: https://docs.cognee.ai/core-concepts/further-concepts/ontologies

[^1_34]: https://www.cognee.ai/blog/deep-dives/ontology-ai-memory

[^1_35]: https://kastle-lab.github.io/assets/publications/2024-LLMs4KGOE.pdf

[^1_36]: https://www.youtube.com/watch?v=ckJpKa8CN5E

[^1_37]: https://arxiv.org/abs/2411.09601

[^1_38]: https://web.stanford.edu/class/cs520/2020/abstracts/shimizu.pdf

[^1_39]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7250591/

[^1_40]: https://ceur-ws.org/Vol-2456/paper65.pdf

[^1_41]: https://www.semantic-web-journal.net/system/files/swj2806.pdf

[^1_42]: https://cyberhillpartners.com/enterprise-ai-ontologies-knowledge-graphs/

[^1_43]: https://www.knowledgegraph.tech/blog/speakers/juan-sequeda/

[^1_44]: https://juansequeda.com

[^1_45]: https://dataengineeringpodcast.com/episodepage/from-academia-to-industry-bridging-data-engineering-challenges

[^1_46]: https://thinklinks.wordpress.com

[^1_47]: https://scholar.google.com/citations?user=R73gJDcAAAAJ\&hl=en

[^1_48]: https://www.knowledgegraph.tech/blog/speakers/anna-fensel/

[^1_49]: https://www.youtube.com/watch?v=Ad3jimQO2zY

[^1_50]: https://sites.google.com/site/annafensel/

[^1_51]: https://www.linkedin.com/posts/helenadeus_questions-irina-calls-them-competency-questions-activity-7403079416966193152-qXn0

[^1_52]: https://www.linkedin.com/posts/helenadeus_to-all-the-knowledge-graph-enthusiasts-whove-activity-7314959660694851584-lrnQ

[^1_53]: https://github.com/getzep/graphiti

[^1_54]: https://www.cognee.ai

[^1_55]: https://comodide.com

[^1_56]: https://comodide.com/tutorial.html

[^1_57]: https://www.falkordb.com/blog/how-to-use-knowledge-graph-tools-for-ai/

[^1_58]: https://docs.ag2.ai/latest/docs/use-cases/notebooks/notebooks/agentchat_graph_rag_falkordb/

[^1_59]: https://www.falkordb.com/blog/how-to-build-a-knowledge-graph/

[^1_60]: https://www.falkordb.com/news-updates/graphrag-sdk-0-5-knowledge-graph-integration/

[^1_61]: https://www.reddit.com/r/GraphRAG/comments/1hwp8i6/knowledge_graph_from_ontology_and_documents_with/

[^1_62]: https://github.com/PinkDot-AI/eschergraph

[^1_63]: https://www.stardog.com/blog/agents-need-democratized-data-too/

[^1_64]: https://www.stardog.com/platform/features/graph-machine-learning/

[^1_65]: https://www.stardog.com/blog/enterprise-ai-requires-the-fusion-of-llm-and-knowledge-graph/

[^1_66]: https://www.stardog.com/blog/stardog-voicebox-faq-how-llm-generative-ai-and-knowledge-graphs-are-the-future-of-data-management/

[^1_67]: https://www.stardog.com/platform/

[^1_68]: https://www.reddit.com/r/LangChain/comments/1lhr4ag/built_an_autonomous_ai_agent_with_langgraph/

[^1_69]: https://www.getzep.com/ai-agents/langgraph-tutorial/

[^1_70]: https://www.freecodecamp.org/news/how-to-build-a-starbucks-ai-agent-with-langchain/

[^1_71]: https://docs.langchain.com/oss/python/langgraph/agentic-rag

[^1_72]: https://www.langchain.com/langgraph

[^1_73]: https://docs.langchain.com/oss/python/langgraph/overview

[^1_74]: https://www.aiengineeringpodcast.com/episodepage/enhancing-ai-retrieval-with-knowledge-graphs-a-deep-dive-into-graphrag

[^1_75]: https://www.stardog.com/blog/designing-llm-applications-with-knowledge-graphs-and-langchain/

[^1_76]: https://aiisc.ai/text2kg2025/

[^1_77]: https://cbp1012.github.io/JOWO-ONTOLLM/

[^1_78]: https://www.knowledgegraph.tech

[^1_79]: https://ijckg2025.github.io

[^1_80]: https://cci.drexel.edu/kgbigdata/2025/

[^1_81]: https://podcasts.apple.com/gb/podcast/knowledge-graph-insights/id1757467395

[^1_82]: https://d3clarity.com/podcast/ontology-vs-taxonomy-ai-knowledge-graphs/

[^1_83]: https://open.spotify.com/show/3CMVAHbd7lNb2yObyl8ERe

[^1_84]: https://www.youtube.com/watch?v=K-y8E7CifDY

[^1_85]: https://www.youtube.com/watch?v=2KVkpUGRtnk

[^1_86]: https://www.youtube.com/watch?v=t-GOSUnI2MI

[^1_87]: https://www.youtube.com/watch?v=0c3WicsmLuo

[^1_88]: https://www.youtube.com/watch?v=XNneh6-eyPg

[^1_89]: https://www.youtube.com/watch?v=LI6jW9x0eOo

[^1_90]: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/your-first-graphrag-demo---a-video-walkthrough/4410246

[^1_91]: https://zitniklab.hms.harvard.edu/projects/KGARevion/

[^1_92]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11491333/

[^1_93]: https://arxiv.org/html/2510.20345v1

[^1_94]: https://sparkco.ai/blog/deep-dive-into-knowledge-graph-reasoning-techniques

[^1_95]: https://www.sciencedirect.com/science/article/pii/S1570826825000022

[^1_96]: https://enterprise-knowledge.com/the-role-of-ontologies-with-llms/

[^1_97]: https://www.youtube.com/watch?v=sygRBjILDn8

[^1_98]: https://blog.opendataproducts.org/knowledge-graphs-context-engineering-and-the-future-of-data-products-88759d540e11

[^1_99]: https://www.linkedin.com/posts/jeremyravenel_how-can-an-ontology-be-integrated-into-a-activity-7294852298965483520-vhsb

[^1_100]: https://support.noduslabs.com/hc/en-us/articles/21429518472988-Using-Knowledge-Graphs-as-Reasoning-Experts

[^1_101]: https://www.sciencedirect.com/science/article/pii/S1570826825000149

[^1_102]: https://arxiv.org/html/2403.08345v1

[^1_103]: https://arxiv.org/html/2403.03008v1

[^1_104]: https://developer.nvidia.com/blog/insights-techniques-and-evaluation-for-llm-driven-knowledge-graphs/

[^1_105]: https://scale.stanford.edu/ai/repository/knowledge-graphs-context-sources-llm-based-explanations-learning-recommendations

[^1_106]: https://aclanthology.org/2025.emnlp-main.896/

[^1_107]: https://www.bosch-ai.com/research/fields-of-expertise/neuro-symbolic-ai/

[^1_108]: https://www.anthropic.com/research/mapping-mind-language-model

[^1_109]: https://graphrag.com

[^1_110]: https://wordlift.io/blog/en/neuro-symbolic-ai/

[^1_111]: https://arxiv.org/abs/2505.13890

[^1_112]: https://www.ontotext.com/knowledgehub/fundamentals/what-is-graph-rag/

[^1_113]: https://arxiv.org/abs/2302.07200

[^1_114]: https://www.progress.com/resources/videos/introduction-to-graph-rag---enhancing-retrieval-augmented-generation-with-knowledge-graphs

[^1_115]: https://memgraph.com/blog/deep-learning-knowledge-graph

[^1_116]: https://dl.acm.org/doi/10.1145/3746058.3759012

[^1_117]: https://www.reddit.com/r/Rag/comments/1ftgvv4/would_you_always_recommend_knowledge_graph_rag/

[^1_118]: https://www.reddit.com/r/MachineLearning/comments/1egke1v/survey_paper_over_neurosymbolic_ai_with_knowledge/

[^1_119]: https://squirro.com/squirro-blog/ai-agents-inference-knowledge-graphs

[^1_120]: https://blog.metaphacts.com/identifying-causal-relationships-with-knowledge-graphs-and-large-language-models

[^1_121]: https://www.linkedin.com/posts/juansequeda_knowledge-graphs-activity-7225115677320265728-Jey-

[^1_122]: https://www.reddit.com/r/PromptEngineering/comments/1o1lsm8/ai_agent_for_internal_knowledge_documents/

[^1_123]: https://machinelearning.apple.com/research/odke

[^1_124]: https://neurips.cc/virtual/2024/103718

[^1_125]: https://openreview.net/attachment?id=dPRsXPVDbp\&name=pdf

[^1_126]: https://barndoor.ai/ai-tools/cognee-ai/

[^1_127]: https://github.com/zjukg/KG-LLM-Papers

[^1_128]: https://research.monash.edu/en/publications/agent-om-leveraging-llm-agents-for-ontology-matching

[^1_129]: https://www.reddit.com/r/LLMDevs/comments/1iytgp8/mindmap_generator_marshalling_llms_for/

[^1_130]: https://dl.acm.org/doi/10.14778/3712221.3712222

[^1_131]: https://github.com/XiaoxinHe/Awesome-Graph-LLM

[^1_132]: https://arxiv.org/html/2312.00326v21

[^1_133]: https://www.reddit.com/r/ChatGPTCoding/comments/15c3vmi/i_have_spent_that_last_7_months_building_an/

[^1_134]: https://dl.acm.org/doi/10.1016/j.websem.2025.100862

[^1_135]: https://www.sciencedirect.com/science/article/pii/S1570826824000441

[^1_136]: https://scholar.google.com/citations?user=PlBF5eQAAAAJ\&hl=en

[^1_137]: https://groups.google.com/g/ontolog-forum/c/k-DPycxT778/m/0LgHYiQwDAAJ

[^1_138]: https://www.linkedin.com/posts/juansequeda_your-first-knowledge-graph-should-be-about-activity-7403838635277602816-pRR5

[^1_139]: https://cogan-shimizu.github.io

[^1_140]: https://www.semantic-web-journal.net/content/knowledge-engineering-large-language-models-capability-assessment-ontology-evaluation

[^1_141]: https://www.youtube.com/watch?v=HmBXBY_DClo

[^1_142]: https://rave.ohiolink.edu/etdc/view?acc_num=wright1503504081751496

[^1_143]: https://www.emergentmind.com/articles/2411.09601

[^1_144]: https://arxiv.org/abs/2311.07509

[^1_145]: https://www.youtube.com/watch?v=r09tJfON6kE

[^1_146]: https://www.youtube.com/watch?v=6_zd535EyCk

[^1_147]: https://www.youtube.com/watch?v=cZHkD5RiGyo

[^1_148]: https://www.youtube.com/watch?v=0oDgruiW7Gw

[^1_149]: https://www.microsoft.com/en-us/research/project/graphrag/videos/

[^1_150]: https://knowledgegraphinsights.com/chris-mungall/

[^1_151]: https://www.ontoforce.com/knowledge-graph/ontology

[^1_152]: https://www.nature.com/articles/s41597-024-03171-w

[^1_153]: https://www.falkordb.com

[^1_154]: https://www.youtube.com/watch?v=J5aar9j9FWQ

[^1_155]: https://help.getzep.com/graphiti/getting-started/welcome

[^1_156]: https://www.facebook.com/groups/techtitansgroup/posts/1525512875442692/

[^1_157]: https://www.research.ed.ac.uk/en/publications/neurosymbolic-ai-for-reasoning-over-knowledge-graphs-a-survey/

[^1_158]: https://2024.eswc-conferences.org/wp-content/uploads/2024/05/77770011.pdf

[^1_159]: https://uk.sagepub.com/sites/default/files/upm-assets/147311_book_item_147311.pdf

[^1_160]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425032580

[^1_161]: https://www.scribd.com/document/920972690/An-Introduction-to-Knowledge-Graphs

[^1_162]: https://drops.dagstuhl.de/storage/04dagstuhl-reports/volume12/issue09/22372/DagRep.12.9.60/DagRep.12.9.60.pdf

[^1_163]: https://arxiv.org/abs/2104.12622

[^1_164]: https://www.semantic-web-journal.net/content/neuro-symbolic-system-over-knowledge-graphs-link-prediction-0

[^1_165]: https://philarchive.org/rec/ALLABF-2

[^1_166]: https://www.semanticscholar.org/paper/CoModIDE-The-Comprehensive-Modular-Ontology-IDE-Shimizu-Hammar/0eebc85eb7072f2a5512ec6d8210afb4a2669df1

[^1_167]: https://www.stardog.com/blog/what-llms-dont-know/

[^1_168]: https://duplocloud.com/blog/langchain-vs-langgraph/

[^1_169]: https://www.sciencedirect.com/science/article/pii/S1570826825000034

