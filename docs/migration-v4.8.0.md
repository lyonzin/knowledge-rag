# Migration Guide: v4.7.1 → v4.8.0

## TL;DR

- **`pip install -U knowledge-rag`** — comportamento idêntico à v4.7.1 se você não tocar em `config.yaml`. Zero ação necessária.
- **Opt-in features** requerem edits em `config.yaml` + reindex.
- **Nenhum breaking change.** Assinaturas das MCP tools preservadas byte-for-byte.
- **Ganho principal (multilingual)**: +1.88pp Recall@10 medido no benchmark A/B do repo (p<0.001) e ~+4pp em queries PT-BR.

## O que é opt-in

### 1. Multilingual embeddings

```yaml
# config.yaml
models:
  embedding:
    profile: "multilingual"  # intfloat/multilingual-e5-large, 1024D
```

Depois:

```python
reindex_documents(force=True)
```

Reindex é **obrigatório** — o modelo `multilingual-e5-large` exige prefixes (`"passage: "` para chunks indexados, `"query: "` para queries). Chunks embedados antes da mudança de profile ficariam sem prefix e a similarity degradaria silenciosamente.

**Tempo estimado (corpus 3000-4000 docs):**

| Hardware | Tempo | Observação |
|---|---|---|
| GPU (RTX 3080 Ti, CUDA 12) | ~5-8 min | probe `verify_gpu_readiness()` valida stack antes |
| CPU | ~40-60 min | ONNX serializa inference — parallel_workers não acelera embedding |

Durante o reindex, o RAG continua servindo queries da collection antiga graças ao **dual-collection swap** (F5). Zero downtime.

### 2. GPU auto-detect (default novo)

**Antes de v4.8.0**: `gpu: false` (opt-in explícito).
**Depois**: `gpu: "auto"` (novo default) — probe CUDA no startup, usa se todos os 4 checks passarem, cai pra CPU silently se qualquer um falhar.

Os 4 checks executados por `FastEmbedEmbeddings.verify_gpu_readiness()`:

1. `CUDAExecutionProvider` presente em `onnxruntime.get_available_providers()`
2. DLLs NVIDIA reachable via `PATH`: `cudart64_12.dll`, `cudnn64_9.dll`, `cublasLt64_12.dll` (Windows) ou `.so.12`/`.so.9` equivalentes (Linux)
3. `nvidia-smi --query-gpu=name,memory.total` exit 0
4. Minimal single-op `InferenceSession` inicializa no provider CUDA

Se você tinha `gpu: false` explícito no config: comportamento preservado byte-for-byte.
Se você NÃO tinha `gpu:` no config: v4.8.0 vai probar CUDA no startup (~500ms-2s overhead em máquinas sem GPU). Pra desligar completamente: `gpu: false`.

**Full dependency chain** para CUDA 12 (deve estar tudo no mesmo venv):
- `onnxruntime-gpu` (CUDA 12 variant, via `--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`)
- `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, siblings

Troubleshooting dos 6 modos de falha mais comuns em `docs/gpu-setup.md`.

### 3. Batch + parallel indexing

```yaml
# config.yaml
documents:
  batch_size: 500          # v4.7.1 default preservado
  parallel_workers: 4      # opt-in, default 1 (single-thread)
```

O ganho vem de **SQLite writes overlapping com a próxima batch de ONNX inference** — a embedding em si é serializada pelo lock da ONNX session (nunca há inference paralela). Windows: monitorar estabilidade em `parallel_workers > 4` (extra `[WARN]` no startup por conta do threading + SQLite contention).

Documentos com uma única batch fazem short-circuit no thread pool — nenhum overhead de orquestração se não vale a pena.

### 4. `max_results` default subiu 20 → 100

Callers que passavam `max_results=N` explícito na chamada MCP: **não afetados**.
Callers que NÃO passavam recebiam antes 20 (silent clamp), agora recebem 100.

Isso NÃO é breaking — é fix da assimetria do candidate pool em hybrid retrieval: BM25 puxava até `max_results * 20 = 400` candidatos enquanto semantic era capado em `min(max_results * 3, config.max_results) = min(15, 20) = 20`. Semantic silently starved sem qualquer log signal, degradando quality em qualquer query onde o branch semantic tinha algo melhor a contribuir.

### 5. Resume interrupted reindex

```python
reindex_documents(force=True, resume=True)  # smart reindex apenas
```

Recupera de `data/reindex_checkpoint.json` (auto-escrito a cada 500 docs OU 30s, o que vier primeiro).

Checkpoint armazena: `indexed_doc_ids`, `chunks_processed`, `config_signature` (SHA256 de `embedding_model | embedding_dim | chunk_size | chunk_overlap`).

Se `config_signature` mudou entre checkpoint write e resume load → checkpoint invalidado com WARN. Isso previne collection mista (parte com vetores do modelo antigo, parte com vetores do modelo novo) que degradaria retrieval silenciosamente.

Combos rejeitados:
- `resume=True + full_rebuild=True` — nuclear rebuild wipe collection primeiro; resume em cima disso não faria sentido. Use `nuclear_rebuild(swap=True)` sync se quiser zero-downtime full rebuild.
- `resume=True + force=False` — `resume=True` implicitamente força `mode='smart_reindex'` para não cair silenciosamente em `incremental` (que ignoraria o checkpoint).

Missing/corrupt/future-version/non-dict checkpoints → degrade gracioso pra "start fresh". Zero raise.

## Novo comportamento visível (sem opt-in)

### Rebuild não derruba mais o RAG

**v4.7.1**: `nuclear_rebuild()` deletava a collection primeiro → queries retornavam vazio por 4min-40h (dependendo do corpus + hardware).

**v4.8.0**: `nuclear_rebuild(swap=True)` (novo default) cria staging collection timestamped → popula → valida três gates (`count >= baseline * 0.9`, 4 de 5 queries canônicas retornam hits, `query()` não raise) → atomic two-step swap via `Collection.modify(name=)` → delete old collection.

Queries continuam servindo da collection antiga durante toda a populate/validate phase. Janela de risco no swap em si: microssegundos entre `rename staging → prod_new` e `delete prod_old`. Se falhar aí, `_cleanup_stale_staging_collections()` do próximo boot resolve orphans (>24h TTL).

Legacy destrutivo preservado como `nuclear_rebuild(swap=False)` pra backwards compat / forced-cleanup edge cases.

**Storage**: 2x temporário durante rebuild (prod + staging). Volta pra 1x após swap. Staging órfã (>24h) limpa auto no boot.

### Progress + ETA em `get_reindex_status()`

Novos campos (durante active reindex):
- `chunks_processed` — chunks committed a ChromaDB até agora
- `chunks_total` — rolling estimate (0 durante warmup, depois running average `completed_docs × total_files`)
- `throughput_cps` — chunks/sec, sliding window bounded por min(100 samples, 30s)
- `eta_seconds` — derivado de throughput + remaining chunks, guarded contra div-by-zero
- `checkpoint_saved_at` — ISO timestamp do último checkpoint write (`null` até 500 docs / 30s)
- `resumed` — bool, `true` se o run atual foi recuperado via `resume=True`

Idle path (`active: false`) inalterado — ainda retorna só `last_result` ou `last_error`.

## Security (v4.8.0 Fase 0.5)

Sete vetores fechados que eram latent gaps desde v4.5.1 (as primitives em `mcp_server/security.py` existiam, mas ninguém tinha wireado nos callers):

| Vetor | Fix |
|---|---|
| Path traversal em 5 tools (`get_document`, `add_document_from_content`, `update_document_content`, `remove_document_by_path`, `search_similar`) | `validate_path_within` na entrada; retorno estruturado `{"error": "Filepath rejected: ..."}` |
| Prompt injection via `add_from_url` (OWASP LLM01:2025) | `sanitize_external_content` no body fetched; provenance envelope + defuse de sentinels (`<|im_start|>`, `[INST]`, etc.) |
| Auth bypass em HTTP transports (`sse`, `streamable-http`) | `BearerAuthMiddleware` na ASGI app do FastMCP quando `config.auth_bearer_token` set; token unset → `[WARN]` + comportamento aberto legacy |

Canal de report privado: **GitHub Security Advisory form** — `/security/advisories/new` (Private Vulnerability Reporting habilitado no repo). `SECURITY.md` atualizado.

## Docs novas nesta release

- `docs/gpu-setup.md` — 4 checks executados + full dependency chain + troubleshooting dos 6 modos de falha mais comuns
- `docs/reindex-operations.md` — sync vs async gotcha (`full_rebuild=True` daemon thread morre com o processo), swap workflow, resume semantic
- `docs/refactor-audit-v4.8.0-fase-6.md` — audit da F6 (funções tocadas + LOC pré/pós refactor)
- `docs/perf-baseline-v4.7.1.md` — reference microbenchmarks do tip de v4.7.1

## FAQ

**Q: Preciso reindexar sempre?**
A: Só se mudar `profile`, `query_prefix` ou `passage_prefix`. Chunks antigos foram embedados sem esses valores — similarity ficaria degradada.

**Q: `nuclear_rebuild(swap=True)` é atômico de verdade?**
A: A troca em si sim (single `Collection.modify()` call por lado). A janela de risco é microssegundos entre `rename staging → prod_new` e `delete prod_old`. Se o processo morrer nesse ponto exato, `_cleanup_stale_staging_collections()` do próximo boot resolve orphans.

**Q: E se o processo Python morrer durante `nuclear_rebuild(swap=True)`?**
A: Staging fica órfã (nome pattern `{collection_name}__staging_{unix_ts}`). Produção continua íntegra. Próximo boot limpa staging >24h auto. Zero corrupção. Já mid-populate: prod nunca foi tocada, staging fica pra ser limpa.

**Q: Como sei se o profile mudou entre um restart e outro?**
A: `get_index_stats()` retorna `embedding_model` + `embedding_dim`. Compare com config atual. F1 adicionou log quando profile ≠ metadata do chunk.

**Q: `parallel_workers` > 4 é seguro?**
A: Em Linux geralmente sim. Em Windows a combinação threading + SQLite contention pode causar instabilidade — v4.8.0 emite `[WARN]` no startup se `workers > 4` em Windows. Se você não vê ganho mensurável de `4 → 8`, ficar em 4 é mais seguro.

## Rollback

Se algo quebrar após v4.8.0:

1. `pip install knowledge-rag==4.7.1` — reinstala a versão anterior.
2. **Se você reindexou com `multilingual`**: chunks ficam com 1024D, v4.7.1 espera 384D → dimension mismatch no primeiro query. Solução: rode `reindex_documents(force=True, full_rebuild=True)` com config v4.7.1 (bge-small-en-v1.5) para reconstruir a collection em 384D.
3. **Se você trocou pra `nuclear_rebuild(swap=True)` mid-rebuild e voltou pra v4.7.1**: staging órfã fica no disco (v4.7.1 não tem o cleanup). Delete manual:
   ```python
   from chromadb import PersistentClient
   client = PersistentClient(path="data/chromadb")
   for col in client.list_collections():
       if "__staging_" in col.name:
           client.delete_collection(name=col.name)
   ```
4. **Se você tinha `gpu: "auto"` implícito e quer reproducibility exata**: adicione `gpu: false` explícito antes do rollback.

## Referência dos PRs (7 fases)

| Fase | PR | Escopo |
|---|---|---|
| 0.5 | #147 | Security wire (path traversal + prompt injection + bearer auth) + BM25 unicode tokenizer |
| 1 | #148 | Embedding profile + query/passage prefix |
| 2 | #149 | GPU tri-state auto-detect + banner + probe |
| 3 | #150 | Batch/parallel indexing + max_results asymmetry fix |
| 4 | #151 | Reindex checkpoint + resume + granular progress |
| 5 | #152 | Zero-downtime nuclear_rebuild via staging swap |
| 6 | #153 | Hotspot decomposition (40+ helpers, complexity D→C) |
