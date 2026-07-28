# ADR-0001 — Fase 1: Security Hardening & Quick Wins

**Status:** Accepted
**Date:** 2026-07-27
**Author:** Ailton Rocha (Lyon.)
**Deciders:** Ailton
**Supersedes:** —
**Superseded by:** —

---

## Context

Auditoria independente do estado atual do `knowledge-rag` v4.5.0 (documentada em `scratchpad/knowledge-rag-audit.md`) identificou:

**Débitos críticos de segurança:**
- Path traversal potencial em `add_document` / `update_document` / `remove_document` (`server.py:1883`) — `config.documents_dir / filepath` sem sanitização de `..`.
- Symlink escape em `ingestion.py:279` — `os.walk(followlinks=True)` sem verificar se destino escapa `documents_dir`.
- Bearer token declarado em `config.py:602-606` mas enforcement não localizado no FastMCP call — precisa auditoria.
- **Zero defesa contra prompt injection** em `add_from_url` e `parse_file` — vetor real de ataque em RAG-over-untrusted-content.

**Débitos de qualidade não-críticos:**
- Version drift: `server.py:22-23` diz `Versao: 3.5.2` mas pyproject.toml é `4.5.0`.
- `Precision@5` mentida no docstring de `evaluate_retrieval` (`server.py:2103`).
- Truncation silenciosa em outputs de retrieval (LLM cliente lê silêncio como ausência).
- Query stopwords não filtradas — "how does X work" polui BM25.
- Sem CLI direto de query/stats/list — user precisa passar por MCP client.
- README de 80 KB monolítico — falta docs modular.

## Decision

Executar **Fase 1** do roadmap `graphify-vs-knowledge-rag-FINAL.md` numa única branch feature (`feature/fase1-security-hardening`), com 8 tasks agrupadas em 3 blocos de execução paralela.

### Bloco A — Security Hardening (bloqueante, prioridade máxima)

| ID | Descrição | DoD |
|---|---|---|
| Q1.1 | Path traversal fix + symlink escape + bearer auth audit | Testes de regressão que provam path escape falha; symlink escape falha; bearer token bloqueia request sem header |
| Q1.2 | Defesa 3-camadas contra prompt injection (wrap + sentinel neutralization + evidence marker) | Testes com corpus hostil provam sanitização; SECURITY.md atualizado com threat model |

### Bloco B — Quality Fixes (independentes, paralelizáveis)

| ID | Descrição | DoD |
|---|---|---|
| Q1.3 | Query stopwords multi-idioma (EN/PT/ES/DE/FR/IT) | Teste que verifica filtragem correta; documentado em config.yaml |
| Q1.5 | Version drift fix (docstring + README) | Pre-commit hook valida sincronia; `__version__` importado |
| Q1.6 | Truncation-aware output com seed protection | Teste que verifica aviso `[!] TRUNCATED` presente + top-1 preservado |
| Q1.8 | Precision@5 implementation | Teste que valida P@5 = hits_at_5 / 5 |

### Bloco C — DX / UX (independente)

| ID | Descrição | DoD |
|---|---|---|
| Q1.4 | CLI subcommands: search / stats / list / add | `knowledge-rag search "..."` funciona standalone; help messages; testes de integração |

### Bloco D — Docs (executa após A+B+C completos)

| ID | Descrição | DoD |
|---|---|---|
| Q1.7 | Modularizar README em `docs/` | README reduzido ≤15KB; docs/architecture.md + deployment.md + tutorial.md + api-reference.md + troubleshooting.md + security.md + benchmarks.md separados |

---

## Consequences

### Positive

- **Fecha 2 CVE-classes**: path traversal (CWE-22) e symlink escape (CWE-59) — atualmente knowledge-rag é vulnerável.
- **Fecha 1 attack surface inteira**: prompt injection em conteúdo externo (LLM01:2025 OWASP Top 10 for LLMs).
- **Melhora percepção de trust** — SECURITY.md atualizado + threat model publicado + release destaca "Security Hardening".
- **Reduz suporte**: CLI direto elimina questões "como faço query fora do Claude Code".
- **Prepara terreno para Fase 2** (refactor arquitetural) — código mais seguro reduz risco de regressão durante refactor.

### Neutral

- Version bump: **v4.5.0 → v4.6.0** (feature + security = minor bump segundo semver).
- CHANGELOG entry obrigatória (gated por `quality-gate.yml`).
- Test count vai subir (baseline `test-count-baseline.txt` precisa atualizar).

### Negative / Risks

1. **Path traversal fix pode quebrar workflows existentes** que dependem de `..` em paths.
   - Mitigação: adicionar flag `--allow-outside` opcional com WARN log; documentar breaking change no CHANGELOG.

2. **Prompt injection defense pode gerar false positives** em docs legítimos que contêm markdown "### system:".
   - Mitigação: aplicar sanitização APENAS em conteúdo vindo de fontes externas (`add_from_url`), não em docs internos indexados.

3. **CLI subcommands podem conflitar com scripts do user** que já usam `knowledge-rag` como `knowledge-rag <PID>` ou similar.
   - Mitigação: subcommands são adição, não substituição; `knowledge-rag` sem args continua sendo `server`.

4. **Refactor de docstrings/README pode gerar merge conflicts** com PRs paralelos.
   - Mitigação: Q1.7 executa POR ÚLTIMO na fase, permitindo outros PRs mergearem primeiro.

---

## Definition of Done (DoD) — Fase 1 como um todo

Antes de merge para master, TODOS os itens abaixo devem passar:

- [ ] Todas as 8 tasks completas
- [ ] `pytest tests/` verde localmente (todos os 25+ testes existentes passam)
- [ ] Testes novos adicionados para cada task com DoD específico
- [ ] Coverage não degrada abaixo de 35% (`fail_under = 35` em pyproject.toml)
- [ ] `ruff check` + `ruff format --check` limpos
- [ ] `mypy strict` limpo nos módulos anotados
- [ ] `bandit -r mcp_server/` sem HIGH severity
- [ ] `pip-audit --strict` sem CRITICAL
- [ ] `gitleaks detect --config .github/gitleaks.toml --no-git` limpo
- [ ] `docs/architecture.md`, `docs/security.md`, `docs/tutorial.md` criados (mínimo)
- [ ] `SECURITY.md` atualizado com threat model tabela vector×mitigation
- [ ] `CHANGELOG.md` com seção `[4.6.0] - YYYY-MM-DD` documentando:
  - Security: 3 fixes de segurança
  - Added: 4 features (stopwords, truncation-aware, Precision@5, CLI subcommands)
  - Fixed: 2 issues (version drift, docs mentida)
- [ ] Version bump `pyproject.toml` 4.5.0 → 4.6.0 + `__init__.py` sincronizado
- [ ] `test-count-baseline.txt` atualizado
- [ ] PoC scripts em `tests/security/` provando fixes:
  - `test_security_path_traversal.py` — provas de escape bloqueado
  - `test_security_symlink_escape.py` — provas de walk protegido
  - `test_security_prompt_injection.py` — provas de sanitização
  - `test_security_bearer_auth.py` — provas de enforcement

---

## Execution Plan

**Branch:** `feature/fase1-security-hardening` (criada em 2026-07-27).

**Sequenciamento:**
1. Bloco A (Security) + Bloco B (Quality) + Bloco C (CLI) executam em **paralelo**.
2. Bloco D (Docs) executa **após** merge dos blocos A+B+C.
3. Cada task = 1 commit atômico (facilita revert/bisect).
4. Testes rodam localmente antes de cada commit.
5. Push + PR ao final da fase, com `gh pr create` documentando cada bloco.
6. Merge squash para master (mantém history linear).

**Roles (dispatch DEV SQUAD):**
- **GUARD (AppSec)** — dono Bloco A (Q1.1 + Q1.2).
- **python-pro** — dono Bloco B (Q1.3 + Q1.5 + Q1.6 + Q1.8).
- **python-pro** — dono Bloco C (Q1.4).
- **VANGUARD (self)** — dono Bloco D (Q1.7) + coordenação + revisão de arquitetura + merge.
- **VERITAS (QA)** — reviewer final antes de merge (test coverage + regressão).

**Timing estimado:** 1-2 semanas solo, 2-4 dias com dispatch paralelo.

---

## References

- Reverse engineering completo do Graphify: `scratchpad/graphify-analysis.md`
- Auditoria estado real knowledge-rag: `scratchpad/knowledge-rag-audit.md`
- Roadmap consolidado: `scratchpad/graphify-vs-knowledge-rag-FINAL.md`
- OWASP LLM01:2025 — Prompt Injection
- CWE-22 — Improper Limitation of a Pathname to a Restricted Directory
- CWE-59 — Improper Link Resolution Before File Access
- CWE-287 — Improper Authentication
- Graphify prompt injection defense: `graphify/llm.py:522-718` (referência inspiradora, não código copiado — reescrito para MIT/context knowledge-rag)
