# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ontos is a research project exploring how ontologies, mind maps, and knowledge graphs can enhance LLM reasoning and context engineering. The focus areas include:

- Knowledge graph-enhanced RAG (GraphRAG) for improved accuracy over vector-only retrieval
- Temporal knowledge graphs for AI agent memory
- Neurosymbolic AI combining symbolic reasoning with neural learning
- Multi-agent systems that build and reason over shared knowledge graphs

## Repository Structure

This is a research/documentation repository with no code. Contents:
- `README.md` - Project description and research goals
- `Initial_research_of_sources_prplx_labs.md` - Comprehensive survey covering projects, researchers, tools, conferences, and benchmarks
- `Initial_research_by_Claude.md` - Supplementary research on underrepresented topics (knowledge editing, GNN+LLM, commonsense KBs, causal reasoning)
- `palantir_foundry_study.md` - Notes on Palantir Foundry's ontology-based architecture
- `NotebookLM Mind Map.png` - Visual mind map of concepts

## Key Concepts

**GraphRAG**: Combines knowledge graph structure with LLM generation. Microsoft's approach uses community detection and hierarchical summarization. Benchmarks show 35-80% accuracy improvements over vector-only RAG.

**Temporal Knowledge Graphs**: Zep's Graphiti framework tracks temporal validity of facts, enabling agent memory that handles evolving information.

**Neurosymbolic AI**: Hybrid systems combining symbolic reasoning (ontologies, logic) with neural learning. Key approaches include Logic Tensor Networks, DeepProbLog, and Scallop.

**Key Tools**:
- Graphiti (Zep) - Temporal KG for agent memory
- Cognee - Memory-augmented LLM with RDF/OWL support
- MindMap - KG prompting for transparent reasoning
- CoModIDE - Modular ontology engineering IDE
- ROME/MEMIT - Knowledge editing in LLMs

**Key Researchers**: Pascal Hitzler, Cogan Shimizu, Juan Sequeda, Paul Groth, Anna Fensel, Helena Deus, Denny Vrandečić (Wikidata)

## Research Assistance

When asked to find resources in this space, prioritize:
1. Academic papers from ACL, EMNLP, ISWC, ESWC, KGC conferences
2. Open-source implementations on GitHub
3. Industry blogs from Neo4j, Stardog, Zep, Anthropic, Microsoft Research
4. Podcasts: Knowledge Graph Insights, Catalog and Cocktails
