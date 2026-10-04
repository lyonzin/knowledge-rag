# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║              KNOWLEDGE RAG — Software Developer Preset                       ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Optimized for: Full-stack development, APIs, DevOps, cloud infrastructure,
#                system design, architecture decisions, and technical docs.
#
# Usage:  cp presets/developer.yaml config.yaml


# ============================================================================
# PATHS
# ============================================================================

paths:
  documents_dir: "./documents"
  data_dir: "./data"
  models_cache_dir: "./models_cache"


# ============================================================================
# DOCUMENTS
# ============================================================================

documents:
  supported_formats:
    - .md        # READMEs, ADRs, design docs
    - .txt       # Plain text notes
    - .pdf       # Technical papers, specs
    - .docx      # Word documents
    - .py        # Python source code
    - .c         # C source code
    - .h         # C/C++ headers
    - .cpp       # C++ source code
    - .js        # JavaScript
    - .jsx       # React JSX
    - .ts        # TypeScript
    - .tsx       # React TypeScript TSX
    - .json      # API schemas, configs
    - .xml       # XML config files
    - .csv       # Data files, logs
    - .ipynb     # Jupyter Notebooks

  exclude_patterns:
    - "node_modules"
    - ".venv"
    - "__pycache__"
    - ".git"
    - "dist"
    - "build"
    - ".next"

  chunking:
    chunk_size: 1200      # Slightly larger for code + prose docs
    chunk_overlap: 250


# ============================================================================
# MODELS
# ============================================================================

models:
  embedding:
    model: "BAAI/bge-small-en-v1.5"
    dimensions: 384

  reranker:
    enabled: true
    model: "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k_multiplier: 3


# ============================================================================
# SEARCH
# ============================================================================

search:
  default_results: 5
  max_results: 20
  collection_name: "knowledge_base"


# ============================================================================
# CATEGORIES
# ============================================================================
# Suggested folder structure:
#   documents/
#   ├── architecture/       → "architecture"  (Design docs, ADRs, diagrams)
#   ├── api/                → "api"           (OpenAPI specs, API guides)
#   ├── runbooks/           → "runbooks"      (Incident response, deploy guides)
#   ├── backend/            → "backend"       (Server-side code & docs)
#   ├── frontend/           → "frontend"      (UI/UX, components, styles)
#   ├── infrastructure/     → "infra"         (Terraform, K8s, CI/CD)
#   ├── database/           → "database"      (Schemas, migrations, queries)
#   ├── testing/            → "testing"       (Test strategies, frameworks)
#   └── notes/              → "notes"         (Meeting notes, RFCs, brainstorms)

category_mappings:
  "architecture": "architecture"
  "api": "api"
  "runbooks": "runbooks"
  "backend": "backend"
  "frontend": "frontend"
  "infrastructure": "infra"
  "database": "database"
  "testing": "testing"
  "notes": "notes"


# ============================================================================
# KEYWORD ROUTING
# ============================================================================

keyword_routes:

  # --- Architecture & Design ---
  architecture:
    - architecture
    - system design
    - design doc
    - design document
    - adr
    - architecture decision
    - trade-off
    - scalability
    - microservice
    - monolith
    - event-driven
    - domain-driven
    - ddd
    - cqrs
    - event sourcing
    - saga pattern

  # --- API Development ---
  api:
    - api
    - rest
    - graphql
    - grpc
    - openapi
    - swagger
    - endpoint
    - webhook
    - rate limit
    - pagination
    - versioning
    - oauth
    - jwt
    - bearer token
    - cors

  # --- Operations & Runbooks ---
  runbooks:
    - runbook
    - incident
    - outage
    - rollback
    - deploy
    - deployment
    - hotfix
    - postmortem
    - on-call
    - pagerduty
    - alert
    - sla
    - slo
    - sli
    - error budget
    - uptime

  # --- Backend ---
  backend:
    - backend
    - server
    - middleware
    - authentication
    - authorization
    - session
    - cache
    - queue
    - worker
    - cron job
    - background job
    - websocket
    - sse

  # --- Frontend ---
  frontend:
    - frontend
    - react
    - nextjs
    - vue
    - angular
    - svelte
    - component
    - hook
    - state management
    - redux
    - zustand
    - css
    - tailwind
    - responsive
    - accessibility
    - a11y
    - ssr
    - ssg
    - hydration

  # --- Infrastructure & DevOps ---
  infra:
    - terraform
    - kubernetes
    - k8s
    - docker
    - container
    - helm
    - ci/cd
    - github actions
    - gitlab ci
    - jenkins
    - aws
    - gcp
    - azure
    - cloud
    - load balancer
    - cdn
    - nginx
    - reverse proxy
    - ssl
    - tls
    - dns

  # --- Database ---
  database:
    - database
    - sql
    - postgres
    - postgresql
    - mysql
    - mongodb
    - redis
    - elasticsearch
    - migration
    - schema
    - index
    - query optimization
    - n+1
    - connection pool
    - replication
    - sharding
    - orm
    - prisma
    - typeorm
    - sqlalchemy

  # --- Testing ---
  testing:
    - test
    - testing
    - unit test
    - integration test
    - e2e
    - end-to-end
    - jest
    - pytest
    - vitest
    - cypress
    - playwright
    - mock
    - stub
    - fixture
    - coverage
    - tdd
    - bdd


# ============================================================================
# QUERY EXPANSION
# ============================================================================

query_expansions:

  # --- Languages & Runtimes ---
  js:                 ["javascript", "js", "ecmascript"]
  javascript:         ["javascript", "js", "ecmascript"]
  ts:                 ["typescript", "ts"]
  typescript:         ["typescript", "ts"]
  py:                 ["python", "py"]
  python:             ["python", "py"]
  rb:                 ["ruby", "rb"]
  go:                 ["golang", "go"]
  golang:             ["golang", "go"]
  rs:                 ["rust", "rs"]
  rust:               ["rust", "rs"]
  cs:                 ["csharp", "c#", "cs", "dotnet"]
  csharp:             ["csharp", "c#", "cs", "dotnet"]

  # --- Frameworks ---
  next:               ["nextjs", "next.js", "next"]
  nextjs:             ["nextjs", "next.js", "next"]
  express:            ["expressjs", "express.js", "express"]
  fastapi:            ["fastapi", "fast api"]
  django:             ["django", "drf", "django rest framework"]
  rails:              ["ruby on rails", "rails", "ror"]
  spring:             ["spring boot", "spring", "spring framework"]
  laravel:            ["laravel", "php"]

  # --- Infrastructure ---
  k8s:                ["kubernetes", "k8s"]
  kubernetes:         ["kubernetes", "k8s"]
  tf:                 ["terraform", "tf", "iac"]
  terraform:          ["terraform", "tf", "infrastructure as code"]
  docker:             ["docker", "container", "containerization"]
  ci:                 ["continuous integration", "ci", "pipeline"]
  cd:                 ["continuous deployment", "continuous delivery", "cd"]
  cicd:               ["ci/cd", "cicd", "continuous integration", "continuous deployment"]
  gh actions:         ["github actions", "gh actions", "gha"]

  # --- Databases ---
  pg:                 ["postgresql", "postgres", "pg"]
  postgres:           ["postgresql", "postgres", "pg"]
  postgresql:         ["postgresql", "postgres", "pg"]
  mongo:              ["mongodb", "mongo"]
  mongodb:            ["mongodb", "mongo"]
  es:                 ["elasticsearch", "elastic", "es"]
  elasticsearch:      ["elasticsearch", "elastic", "es"]
  db:                 ["database", "db"]
  orm:                ["orm", "object relational mapping"]

  # --- Cloud ---
  aws:                ["amazon web services", "aws"]
  gcp:                ["google cloud platform", "gcp", "google cloud"]
  azure:              ["microsoft azure", "azure"]
  s3:                 ["s3", "amazon s3", "object storage"]
  ec2:                ["ec2", "amazon ec2", "compute"]
  lambda:             ["aws lambda", "lambda", "serverless"]
  serverless:         ["serverless", "faas", "function as a service"]

  # --- Architecture Patterns ---
  ddd:                ["domain-driven design", "ddd"]
  cqrs:               ["command query responsibility segregation", "cqrs"]
  ssr:                ["server-side rendering", "ssr"]
  ssg:                ["static site generation", "ssg"]
  spa:                ["single page application", "spa"]
  api:                ["api", "application programming interface"]
  rest:               ["rest", "restful", "rest api"]
  graphql:            ["graphql", "gql"]
  grpc:               ["grpc", "protobuf", "protocol buffers"]

  # --- Observability ---
  apm:                ["application performance monitoring", "apm"]
  sla:                ["service level agreement", "sla"]
  slo:                ["service level objective", "slo"]
  sli:                ["service level indicator", "sli"]
  mttr:               ["mean time to recovery", "mttr"]
  mttf:               ["mean time to failure", "mttf"]

  # --- Testing ---
  tdd:                ["test-driven development", "tdd"]
  bdd:                ["behavior-driven development", "bdd"]
  e2e:                ["end-to-end", "e2e"]
