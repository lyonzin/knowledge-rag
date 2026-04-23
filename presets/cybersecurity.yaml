# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║              KNOWLEDGE RAG — Cybersecurity Preset                            ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Optimized for: Red Team, Blue Team, CTFs, Threat Hunting, Incident Response,
#                Exploit Development, Malware Analysis, OSINT, Compliance.
#
# Usage:  cp presets/cybersecurity.yaml config.yaml
#
# This preset includes:
#   - 8 security-specific categories
#   - 200+ keyword routes across 7 domains
#   - 60+ query expansions (CVE aliases, tool synonyms, attack abbreviations)


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
    - .md
    - .txt
    - .pdf
    - .py       # Exploit scripts, PoCs, tooling
    - .c        # Exploit PoCs, kernel modules
    - .h        # Header files
    - .cpp      # C++ exploit code
    - .json     # Nuclei templates, config dumps
    - .xml      # Nmap XML, Nuclei templates, config files
    - .docx
    - .xlsx
    - .pptx
    - .csv
    - .ipynb    # Jupyter Notebooks (exploit dev, research)

  exclude_patterns: []

  chunking:
    chunk_size: 1000
    chunk_overlap: 200


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
# Folder structure → category mapping.
# Organize your documents/ folder to match these patterns.
#
# Suggested folder structure:
#   documents/
#   ├── security/
#   │   ├── redteam/        → "redteam"
#   │   ├── blueteam/       → "blueteam"
#   │   └── ctf/            → "ctf"
#   ├── aar/                → "aar"  (After Action Reports)
#   ├── logscale/           → "logscale"  (CrowdStrike LogScale / Humio)
#   ├── development/        → "development"
#   └── general/            → "general"

category_mappings:
  "security/redteam": "redteam"
  "security/blueteam": "blueteam"
  "security/ctf": "ctf"
  "security": "security"
  "aar": "aar"
  "logscale": "logscale"
  "development": "development"
  "general": "general"


# ============================================================================
# KEYWORD ROUTING
# ============================================================================
# Security-specific keyword → category routing.
# Queries containing these terms get routed to the matching category.

keyword_routes:

  # --- LogScale / CrowdStrike Query Language ---
  logscale:
    - logscale
    - lql
    - cql
    - humio
    - crowdstrike query
    - formattime
    - groupby
    - base64decode
    - "case{}"
    - regex

  # --- Red Team / Offensive Security ---
  redteam:
    # General offensive
    - pentest
    - exploit
    - payload
    - reverse shell
    - privilege escalation
    - lateral movement
    - c2
    - beacon
    - privesc
    # Frameworks & tools
    - cobalt strike
    - metasploit
    - mimikatz
    - rubeus
    - certipy
    - bloodhound
    - searchsploit
    - exploit-db
    - hashcat
    - hacktricks
    # GTFOBins / LOLBins
    - gtfobins
    - lolbas
    - lolbin
    - suid
    - sudo
    # Active Directory
    - kerberoast
    - dcsync
    - golden ticket
    - pass-the-hash
    - adcs
    # Web attacks
    - sqli
    - xss
    - ssti
    - ssrf
    - lfi
    - rfi
    - xxe
    - deserialization
    - ysoserial
    - upload bypass
    - web shell
    # Windows attacks
    - byovd
    - lol driver
    - lolad
    - lolapps
    - amsi bypass
    - uac bypass
    - potato
    # Evasion & bypass
    - waf bypass
    - hash cracking
    - cve

  # --- Blue Team / Defense ---
  blueteam:
    - detection
    - sigma
    - yara
    - ioc
    - threat hunting
    - incident response
    - forensics
    - malware analysis

  # --- CTF / Wargames ---
  ctf:
    - ctf
    - flag
    - hackthebox
    - htb
    - tryhackme
    - picoctf
    - writeup
    - challenge

  # --- Development ---
  development:
    - python
    - typescript
    - javascript
    - api
    - fastapi
    - django
    - react
    - nodejs

  # --- Security (Anti-bot, WAF, Fingerprinting) ---
  security:
    # Anti-bot & browser automation
    - anti-bot
    - antibot
    - js challenge
    - javascript challenge
    - cdp detection
    - runtime.enable
    - puppeteer
    - playwright
    - selenium
    - nodriver
    - stealth
    - undetected
    # TLS & fingerprinting
    - ja3
    - ja4
    - tls fingerprint
    - fingerprinting
    - curl_cffi
    - got-scraping
    - impersonate
    - http/2 settings
    - browser fingerprint
    - canvas fingerprint
    - webgl fingerprint
    - navigator.webdriver
    - audio context
    - hardware concurrency
    # WAF bypass
    - waf bypass
    - aws waf
    - cloudflare bypass
    - akamai bypass
    - datadome
    - perimeterx
    - imperva bypass
    - 8kb bypass
    - body size limit
    - json sqli
    # Behavioral evasion
    - behavioral
    - mouse movement
    - ghost-cursor
    - humanized
    - flaresolverr
    - turnstile
    - rebrowser
    - botbrowser


# ============================================================================
# QUERY EXPANSION
# ============================================================================
# Security abbreviations, tool aliases, and CVE codenames.
# Ensures "sqli" finds "SQL injection" and vice versa.

query_expansions:

  # --- Web Vulnerabilities ---
  sqli:               ["sql injection", "sqli"]
  sql injection:      ["sql injection", "sqli"]
  xss:                ["cross-site scripting", "xss"]
  cross-site scripting: ["cross-site scripting", "xss"]
  ssrf:               ["server-side request forgery", "ssrf"]
  lfi:                ["local file inclusion", "lfi"]
  rfi:                ["remote file inclusion", "rfi"]
  rce:                ["remote code execution", "rce"]
  xxe:                ["xml external entity", "xxe"]
  ssti:               ["server-side template injection", "ssti"]
  idor:               ["insecure direct object reference", "idor"]
  csrf:               ["cross-site request forgery", "csrf"]
  deserialization:    ["deserialization", "deserialisation", "insecure deserialization"]

  # --- Privilege Escalation ---
  privesc:            ["privilege escalation", "privesc"]
  priv esc:           ["privilege escalation", "privesc"]
  privilege escalation: ["privilege escalation", "privesc"]
  suid:               ["suid", "setuid", "set-uid"]
  potato:             ["potato", "juicypotato", "sweetpotato", "godpotato", "efspotato", "printspoofer"]
  uac:                ["user account control", "uac", "uac bypass"]

  # --- Active Directory ---
  pth:                ["pass-the-hash", "pth"]
  pass-the-hash:      ["pass-the-hash", "pth"]
  dcsync:             ["dcsync", "dc sync", "domain controller sync"]
  kerberoast:         ["kerberoasting", "kerberoast"]
  kerberoasting:      ["kerberoasting", "kerberoast"]
  asrep:              ["as-rep roasting", "asrep", "asreproast"]
  bloodhound:         ["bloodhound", "sharphound"]
  ad:                 ["active directory", "ad"]
  active directory:   ["active directory", "ad"]
  rbcd:               ["resource-based constrained delegation", "rbcd"]
  dpapi:              ["dpapi", "data protection api", "credential manager"]

  # --- Tools ---
  mimikatz:           ["mimikatz", "sekurlsa", "logonpasswords"]
  hashcat:            ["hashcat", "hash cracking", "hash crack"]
  john:               ["john the ripper", "john", "jtr"]
  responder:          ["responder", "llmnr", "nbt-ns", "netbios"]
  volatility:         ["volatility", "memory forensics", "memory analysis"]

  # --- Shells & C2 ---
  revshell:           ["reverse shell", "revshell", "rev shell"]
  reverse shell:      ["reverse shell", "revshell"]
  webshell:           ["web shell", "webshell"]
  web shell:          ["web shell", "webshell"]
  c2:                 ["c2", "command and control", "command-and-control", "beacon"]
  sliver:             ["sliver", "sliver c2"]
  cobalt:             ["cobalt strike", "cobalt", "cs beacon"]

  # --- Protocols ---
  ntlm:               ["ntlm", "net-ntlmv2", "ntlmv2"]
  smb:                ["smb", "server message block", "samba"]
  ldap:               ["ldap", "lightweight directory access protocol"]

  # --- Defense & Detection ---
  waf:                ["web application firewall", "waf"]
  amsi:               ["antimalware scan interface", "amsi", "amsi bypass"]
  defender:           ["windows defender", "defender", "wdfilter"]
  lolbin:             ["lolbin", "lolbas", "living off the land"]
  cron:               ["cron", "crontab", "cronjob", "scheduled task"]
  forensics:          ["forensics", "forensic", "dfir"]
  steganography:      ["steganography", "stego", "steghide"]
  stego:              ["steganography", "stego", "steghide"]
  phishing:           ["phishing", "spearphishing", "social engineering"]

  # --- CVE Codenames ---
  # Maps common vulnerability names to their CVE IDs and related terms.
  printnightmare:     ["printnightmare", "cve-2021-34527", "spoolsv", "printspooler"]
  cve-2021-34527:     ["printnightmare", "cve-2021-34527", "spoolsv"]
  eternalblue:        ["eternalblue", "ms17-010", "smbv1"]
  ms17-010:           ["eternalblue", "ms17-010", "smbv1"]
  pwnkit:             ["pwnkit", "cve-2021-4034", "pkexec"]
  cve-2021-4034:      ["pwnkit", "cve-2021-4034", "pkexec"]
  log4shell:          ["log4shell", "cve-2021-44228", "log4j"]
  cve-2021-44228:     ["log4shell", "cve-2021-44228", "log4j"]
  zerologon:          ["zerologon", "cve-2020-1472", "netlogon"]
  cve-2020-1472:      ["zerologon", "cve-2020-1472", "netlogon"]
  petitpotam:         ["petitpotam", "cve-2021-36942", "efs", "ntlm relay"]
  certifried:         ["certifried", "cve-2022-26923", "adcs"]
  nopac:              ["nopac", "samaccountname", "cve-2021-42278", "cve-2021-42287"]
  proxylogon:         ["proxylogon", "cve-2021-26855", "exchange"]
  proxyshell:         ["proxyshell", "cve-2021-34473", "exchange"]
