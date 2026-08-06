# v4.8.0 Fase 6 — Refactor Hotspot Audit

**Data:** 2026-08-06
**Scope:** funções tocadas em F1–F5 (`8910c35..2e02c1d`) que violam limites Core (função ≤30 linhas, arquivo ≤300 linhas).
**Ferramenta:** `radon cc mcp_server/ -a -nc` + AST scanner + `git blame` cross-ref.

## Metodologia

1. Enumerar todas as funções via AST em `mcp_server/{server,config}.py`.
2. Cruzar com `git blame` filtrando SHAs de F1–F5.
3. Reportar apenas funções >30 linhas com pelo menos uma linha tocada em F1–F5.

## Funções >30 linhas com touch em F1–F5

| Função | Range | Total | Touched | % Touched | Ação |
|---|---|---|---|---|---|
| `KnowledgeOrchestrator._index_all_impl` | server.py L1310-1555 | 246 | 115 | 47% | **REFACTOR** |
| `Config.__post_init__` | config.py L747-931 | 185 | 82 | 44% | **REFACTOR** |
| `reindex_documents` | server.py L3303-3402 | 100 | 50 | 50% | **REFACTOR** |
| `KnowledgeOrchestrator._index_document` | server.py L1562-1652 | 91 | 48 | 53% | **REFACTOR** |
| `KnowledgeOrchestrator._rebuild_via_swap` | server.py L2088-2153 | 66 | 66 | 100% | **REFACTOR** (F5) |
| `KnowledgeOrchestrator._validate_staging` | server.py L1916-1971 | 56 | 56 | 100% | **REFACTOR** (F5) |
| `KnowledgeOrchestrator.reindex_all` | server.py L1726-1778 | 53 | 19 | 36% | LEAVE (marginal) |
| `KnowledgeOrchestrator._rebuild_destructive` | server.py L2035-2086 | 52 | 9 | 17% | LEAVE (touch mínimo) |
| `KnowledgeOrchestrator.__init__` | server.py L1100-1149 | 50 | 16 | 32% | LEAVE (marginal) |
| `get_reindex_status` | server.py L2954-3002 | 49 | 27 | 55% | **REFACTOR** |
| `KnowledgeOrchestrator._cleanup_stale_staging_collections` | server.py L1795-1842 | 48 | 48 | 100% | **REFACTOR** (F5) |
| `FastEmbedEmbeddings._print_gpu_banner` | server.py L420-465 | 46 | 27 | 59% | **REFACTOR** (F2) |
| `KnowledgeOrchestrator.start_reindex_background` | server.py L1668-1713 | 46 | 24 | 52% | **REFACTOR** (F4) |
| `KnowledgeOrchestrator._swap_collections_atomic` | server.py L1973-2015 | 43 | 43 | 100% | **REFACTOR** (F5) |
| `FastEmbedEmbeddings._route_load` | server.py L515-555 | 41 | 41 | 100% | **REFACTOR** (F2) |
| `KnowledgeOrchestrator._load_checkpoint` | server.py L3073-3112 | 40 | 40 | 100% | LEAVE (10 linhas over, split não simplifica) |
| `FastEmbedEmbeddings._embed` | server.py L577-614 | 38 | 19 | 50% | LEAVE (8 linhas over) |
| `KnowledgeOrchestrator.index_all` | server.py L1272-1308 | 37 | 12 | 32% | LEAVE (marginal) |
| `KnowledgeOrchestrator._populate_staging` | server.py L1860-1895 | 36 | 36 | 100% | LEAVE (6 linhas over) |
| `KnowledgeOrchestrator._write_checkpoint` | server.py L3038-3071 | 34 | 34 | 100% | LEAVE (4 linhas over) |
| `FastEmbedEmbeddings._load_model` | server.py L481-513 | 33 | 7 | 21% | LEAVE (F2 já extraiu `_route_load`) |

## Decisões

**11 funções em ação REFACTOR**, priorizadas por tamanho descendente:

1. `_index_all_impl` (246 → target ≤30 no orchestrator, 6–8 helpers)
2. `Config.__post_init__` (185 → split por seção de validação)
3. `reindex_documents` (100 → dispatch por modo)
4. `_index_document` (91 → split parse/embed/store)
5. `_rebuild_via_swap` (66 → orchestrator dos 4 helpers F5)
6. `_validate_staging` (56 → 3 gates → 3 helpers)
7. `_cleanup_stale_staging_collections` (48 → scan + prune helpers)
8. `get_reindex_status` (49 → status por seção)
9. `_print_gpu_banner` (46 → seção por linha)
10. `start_reindex_background` (46 → lock + spawn helpers)
11. `_swap_collections_atomic` (43 → each step = helper)
12. `_route_load` (41 → dispatch por modo)

## Extract module (arquivo ≤300 linhas)

`server.py` = 3938 linhas (>>>300). `config.py` = 935 (>>>300).

**Decisão: DEFERIR extract module pra release futura.** Justificativa:

- `test_backwards_compat.py` valida contrato por AST, e mover classes exige ou (a) re-export em `server.py` (`from .embeddings import FastEmbedEmbeddings`) ou (b) update dos testes. Ambos aumentam superfície de regressão.
- Extract module é multi-hora com risco médio-alto de quebrar imports internos e de tests.
- F6 é oportunístico: entrega valor em splits de função (baixo risco) sem gambling em arquitetura.
- Extract module vira issue separada v4.9.0.

Se `server.py` ficar >4500 linhas na v4.9.0, aí sim escalar prioridade.

## Testes de anti-regressão

Suite completa em chunks:

```bash
pytest tests/test_config.py -v
pytest tests/test_embedding_profile.py tests/test_gpu_auto.py -v
pytest tests/test_batch_parallel.py tests/test_reindex_resume.py -v
pytest tests/test_swap_zero_downtime.py -v
pytest tests/test_backwards_compat.py -v
pytest tests/test_bm25_unicode_tokenizer.py tests/test_bm25_tokenizer_fragment.py -v
pytest tests/test_search.py tests/test_dedup.py -v
pytest tests/test_ingestion.py tests/test_ingestion_property.py -v
pytest tests/test_lazy_embeddings.py tests/test_reranker_fallback.py -v
pytest tests/security/ -v
```

`test_backwards_compat.py` é o gate crítico: contract MCP deve continuar green.
