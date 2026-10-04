# Search scope and retrieval quality

`search_knowledge` can search a subset of the indexed corpus without changing
categories or rebuilding the index. For example:

```json
{
  "query": "How is failover configured?",
  "include_folders": ["manuals/product-a", "runbooks"],
  "exclude_folders": ["manuals/product-a/archive"],
  "max_results": 10,
  "snippet_mode": false
}
```

Both filters are optional lists of directories **relative to `documents_dir`**.
Included directories form a union; exclusions take precedence. Descendants are
included recursively. `manuals` does not match `manuals-old`. Both slash styles
are accepted, and case follows the host filesystem path convention (Windows
case-insensitive; Linux case-sensitive). Empty lists behave like omitted lists.
`.` denotes the corpus root. A missing folder returns no matches; a file path,
absolute path, `..`, Windows alternate data stream, or escaping symlink is an
error. Each list accepts at most 32 paths of at most 4096 characters each.

The scope is applied **before candidate selection** in Chroma, BM25 and FTS5.
It is also part of query-cache identity. Category and folder filters intersect.
Adjacent context remains in each selected document; it cannot pull a passage
from another folder. Filtering operates on indexed documents, so newly added,
moved or edited sources still require normal incremental indexing. Uncached
scoped queries perform source validation and an additional metadata selection;
scoping is not a free latency optimization for very large folders.

Hybrid search retrieves semantic and lexical candidates, fuses their ranks and
optionally reranks a bounded candidate pool. The public `max_results` cap only
limits returned results: it does not truncate the semantic pool before the
reranker can inspect it. When both channels contribute, the pool reserves up
to one result-page from each channel and fills the remaining budget by weighted
fusion. This prevents a lower-weight channel from disappearing entirely before
reranking. `hybrid_alpha=0` and `hybrid_alpha=1` remain explicit single-channel
searches. This correction can change ordering; a lexical-first workload should
be evaluated for regressions rather than assuming a universal accuracy gain.

Evaluate candidate coverage, ranking and evidence coverage separately. Finding
the expected source does not prove that the returned passage answers the
question. Use `snippet_mode=false` for evaluation and `get_document` when the
complete source is required. Normalized result scores are relative to that
result set, not probabilities of correctness or an abstention threshold.

The compact default model was trained for English retrieval. The existing
`multilingual` profile is an explicit alternative using
`intfloat/multilingual-e5-large` with 1024-dimensional vectors and the required
`query: ` / `passage: ` prefixes. Keep the previous index and build a separate
data directory when comparing profiles; even models with equal dimensions are
not interchangeable. See [reindex operations](reindex-operations.md).

The default MiniLM reranker was trained on English MS MARCO, so compare enabled
and disabled reranking on a frozen multilingual evaluation set. Preserve known
failures and use unseen, paired questions when comparing Portuguese and English.
Do not infer general improvement from a small hand-selected sample.

The E5 model uses an ONNX external weights file. If ONNX Runtime rejects a
Hugging Face snapshot because its symlinks resolve outside the model directory,
copy the **complete model cache** to a new directory, following its links while
preserving the Hugging Face cache layout. With the server stopped, this example
copies a relative `models_cache` into a new, previously nonexistent directory:

```bash
python -c "import shutil; shutil.copytree('models_cache', 'models_cache_materialized', symlinks=False)"
```

Then set `paths.models_cache_dir: ./models_cache_materialized` in the isolated
configuration and restart. The copy needs additional disk space and retains
the original cache for rollback. Keep Runtime's external-path validation
enabled. This cache-setting-only workaround was verified with real CPU and
CUDA E5 inference on Windows; it does not require a nonexistent model-path
configuration option. The initial model comparison used a private harness path
override, followed by this separate verification of the public cache setting.

E5 large requires substantially more model memory and computation than BGE
small. Start an isolated comparison with `documents.embed_batch_size: 8` (or
smaller on limited hardware), then measure throughput and actual memory before
raising it. The generic CUDA batch default is designed around smaller models;
it is not a promise that a batch of 256 fits every supported model or GPU.

Model contracts: [E5 model card](https://huggingface.co/intfloat/multilingual-e5-large)
and [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/),
verified 2026-10-04. The model-card benchmarks describe upstream datasets, not
the quality of a user's local corpus.
