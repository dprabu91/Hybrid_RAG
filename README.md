# Hybrid RAG - Knowledge Graph + Vector Search

A powerful **Hybrid Retrieval-Augmented Generation (RAG)** system that combines graph-based knowledge extraction with semantic vector search to answer questions with rich contextual understanding.

## 🎯 Overview

This system processes FAQ documents and builds two complementary knowledge representations:

1. **Knowledge Graph** (Neo4j) - Extracts entities and relationships for structured reasoning
2. **Vector Store** (Embeddings) - Stores semantic representations for similarity-based retrieval

When you ask a question, both approaches are combined to provide comprehensive answers grounded in the original document.

## 🏗️ Architecture

```
FAQ Document (faq.txt)
    ↓
┌─────────────────────────────────────────┐
│   Entity & Relation Extraction (LLM)    │
└─────┬─────────────────────────────┬─────┘
      ↓                             ↓
  Neo4j Graph              Vector Store (JSON)
  - Entities               - Embeddings
  - Relations              - Chunks
      │                             │
      └──────────┬──────────────────┘
                 ↓
          Hybrid Context
                 ↓
            LLM Answer
```

## 📋 Features

- **Automatic Entity Extraction** - Uses GPT-4 to identify entities and their relationships
- **Knowledge Graph Storage** - Maintains structured data in Neo4j
- **Semantic Search** - Similarity-based retrieval using embeddings
- **Hybrid Context** - Combines graph facts and semantic matches for comprehensive answers
- **Interactive Q&A** - Command-line interface for asking questions about your knowledge base

## 🛠️ Requirements

### Dependencies
- Python 3.8+
- `openai` - For LLM and embeddings
- `neo4j` - For graph database
- `python-dotenv` - For environment variables
- `numpy` - For vector operations

### External Services
- OpenAI API key (for GPT-4o-mini and embeddings)
- Neo4j database instance (running locally or remote)

## 📦 Installation

1. **Clone/setup the project**
   ```bash
   cd Hybrid_RAG
   ```
2.  **Create Neo4j Database (Free) **
Go to neo4j.com/cloud/aura-free
Sign up → Create AuraDB Free instance
Save your URI, Username, Password

3. **Install dependencies**
   ```bash
   pip install openai neo4j python-dotenv numpy
   ```

4. **Configure environment variables** - Create `.env` file:
   ```env
   OPENAI_API_KEY=sk-...
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=password
   ```

5. **Prepare your FAQ** - Edit `faq.txt` with your knowledge base content (separate chunks with blank lines)

## 🚀 Usage

### Run the pipeline
```bash
python app.py
```

This will:
1. Load FAQ chunks from `faq.txt`
2. Extract entities and relationships using GPT-4o-mini
3. Store knowledge graph in Neo4j
4. Build vector store with embeddings
5. Start interactive Q&A session

### Example Interaction
```
❓ Question: What is machine learning?

💡 Machine learning is a subset of AI that enables systems 
to learn from data without explicit programming...
```

## 📂 File Structure

| File | Purpose |
|------|---------|
| `app.py` | Main application with all RAG components |
| `faq.txt` | Input knowledge base (FAQ documents) |
| `vector_store.json` | Persisted vector embeddings |
| `.env` | Environment variables (create this) |

## 🔄 Pipeline Stages

### 1. Document Loading
Reads and chunks FAQ text, separating by blank lines.

### 2. Entity Extraction
Uses LLM to extract structured entities and relationships from each chunk:
```json
{
  "entities": [{"name": "AI", "type": "CONCEPT"}],
  "relations": [{"source": "ML", "relation": "SUBSET_OF", "target": "AI"}]
}
```

### 3. Graph Storage
Stores unique entities and relationships in Neo4j knowledge graph.

### 4. Vector Embeddings
- Embeds each chunk using `text-embedding-3-small`
- Stores in `vector_store.json` with original text

### 5. Hybrid Context Retrieval
For each question:
- **Graph Search** - Finds relevant entities and their connections
- **Vector Search** - Finds semantically similar chunks (top-3)

### 6. Answer Generation
Combines both contexts and uses GPT-4o-mini to generate coherent answers.

## ⚙️ Configuration

### Models Used
- **LLM**: `gpt-4o-mini` - Fast extraction and generation
- **Embeddings**: `text-embedding-3-small` - Efficient semantic search
- **Graph DB**: Neo4j - Relationship queries

### Tunable Parameters
In `app.py`:
- `top_k=3` - Number of vector results to include (line ~183)
- `temperature=0.3` - Answer generation temperature (line ~211)
- `temperature=0` - Entity extraction strictness (line ~54, 102)

## 🔍 How It Works

### Knowledge Graph Search
1. Extracts keywords from question
2. Queries Neo4j for matching entities
3. Returns entity types and connected relationships

### Vector Search
1. Embeds the question
2. Calculates cosine similarity with all stored embeddings
3. Returns top-k most similar chunks

### Hybrid Answering
- Prefers graph facts if conflict exists
- Uses semantic context to fill gaps
- Maintains coherent narrative

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenAI API errors | Check `OPENAI_API_KEY` in `.env` |
| Neo4j connection failed | Verify URI, username, password in `.env` |
| JSON parse errors | Check LLM extraction output format |
| Vector store not found | Rebuild by running `app.py` first |

## 📈 Performance Tips

- Start with smaller FAQ (10-50 chunks) for testing
- Increase chunks gradually to avoid API rate limits
- Cache embeddings in `vector_store.json` (only regenerate when FAQ changes)
- Adjust `top_k` based on context window needs

## 🔐 Security

- Keep `.env` file private (add to `.gitignore`)
- Don't commit API keys to version control
- Use restricted Neo4j credentials in production

## 📝 License

This project is provided as-is for educational and commercial use.

---

**Questions?** Check the code comments in `app.py` for detailed implementation notes.

