# RAG Agent — Interrogation de documents PDF en langage naturel

> Agent intelligent basé sur l'approche **Retrieval-Augmented Generation (RAG)** permettant d'interroger des documents PDF via une interface conversationnelle.
---

## Présentation

Ce projet implémente un agent RAG complet capable de :

- **Charger** un ou plusieurs documents PDF (existants ou importés via l'interface)
- **Indexer** le contenu dans une base vectorielle persistante (ChromaDB)
- **Répondre** à des questions en langage naturel en s'appuyant uniquement sur le contenu des documents
- **Citer** les passages sources utilisés pour chaque réponse

L'inférence est réalisée **entièrement en local** grâce à Ollama (aucune clé API externe n'est requise).

---

## Architecture du pipeline RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXATION (offline)                     │
│                                                                 │
│   PDF(s)  ──►  PyMuPDF  ──►  Text Splitter  ──►  Embeddings   │
│                                                (nomic-embed)    │
│                                                      │          │
│                                               ChromaDB (local)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFÉRENCE (online)                       │
│                                                                 │
│  Question  ──►  Embedding  ──►  Similarité  ──►  Top-K chunks  │
│  (user)         (requête)       cosinus         (ChromaDB)      │
│                                                      │          │
│                                              Prompt enrichi     │
│                                                      │          │
│                                           LLM (Ollama / local)  │
│                                                      │          │
│                                              Réponse + sources  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack technique & justification des choix

| Composant | Technologie | Pourquoi ce choix |
|-----------|-------------|-------------------|
| Orchestration RAG | **LangChain** | Framework mature, abstraction des chaînes retrieval + génération, compatibilité multi-modèles |
| Modèle LLM | **Ollama** (Mistral / LLaMA 3) | Inférence 100% locale, pas de coût API, données sensibles protégées |
| Embeddings | **nomic-embed-text** via Ollama | Modèle open-source performant, cohérent avec l'approche full-local |
| Base vectorielle | **ChromaDB** | Légère, persistante sur disque, idéale pour prototypage et démos locales |
| Parsing PDF | **PyMuPDF (fitz)** | Extraction de texte fiable, gestion des layouts complexes |
| Découpage | **RecursiveCharacterTextSplitter** | Découpe respectant la structure sémantique du texte (paragraphes > phrases) |
| Interface | **Gradio** | Déploiement d'interface conversationnelle en quelques lignes, idéal pour démo rapide |
| Conteneurisation | **Docker + Compose** | Reproductibilité de l'environnement, déploiement simplifié sur n'importe quelle machine |
| Configuration | **YAML centralisé** | Paramètres RAG (chunk size, top-k, modèle) modifiables sans toucher au code |

---

## Structure du projet

```
rag-agent/
├── src/                        # Code source principal
│   ├── utils/
│   │   ├── chatbot.py
│   │   ├── config.py
│   │   ├── prepare_vectordb.py 
│   │   ├── ui_settings.py
│   │   └── upload_file.py
│   ├── RAG_GPT_APP.py
│   └── question_response.py
├── data/
│   ├── vectordb/
│   │   ├── processed/          # Chunks indexés (auto-généré)
│   │   └── uploaded/           # PDFs uploadés via l'interface
│   └── images/                 # Assets visuels (icônes Gradio)
│       ├── Agent Icone.jpg
│       ├── ai_agent.png
│       └── user.png
├── config/
│   └── app_config.yml          # Configuration centralisée (modèles, paramètres RAG)
├── Dockerfile                  # Image Docker de l'application
├── docker-compose.yml          # Orchestration multi-services
├── .dockerignore
├── .env                        # Variables d'environnement (non versionné)
├── requirements.txt
└── README.md
```

## Utilisation

1. **Importer un PDF** via l'onglet "Charger un document" ou déposer un fichier dans `data/documents/`
2. **Lancer l'indexation** — le document est découpé, embedé et stocké dans ChromaDB
3. **Poser une question** dans l'interface conversationnelle
4. L'agent retourne une réponse sourcée avec les passages pertinents extraits du document

---

## Améliorations prévues

- [ ] Déploiement sur Hugging Face Spaces
- [ ] Support multi-documents avec filtrage par source
- [ ] Passage à un modèle d'embedding multilingue (Français / Arabe)
- [ ] Ajout d'un historique de conversation (memory)
- [ ] Évaluation automatique de la qualité des réponses (RAGAS)

---

## Auteur

**Oussama Sakini** — Ingénieur IA & Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Oussama%20Sakini-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/oussama-sakini-3b6755273/)
[![Portfolio](https://img.shields.io/badge/Portfolio-oussama--sakini.vercel.app-black?style=flat-square)](https://oussama-sakini.vercel.app)
