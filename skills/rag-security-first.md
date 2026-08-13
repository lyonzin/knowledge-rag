---
name: rag-security-first
description: For any security-related task — threat triage, incident response, MITRE ATT&CK mapping, CVE lookup, exploit analysis, defensive control validation, red/blue/purple team work — always consult the local corpus first. Assumes the RAG is loaded with security content (cybersecurity preset, MITRE data, threat reports, runbooks). Prevents wasted external threat-intel calls and grounds recommendations in the team's actual playbooks.
metadata:
  type: rag-workflow
  kind: domain
  target: any-mcp-client
---

# rag-security-first — RAG-driven security workflow

## When to use this skill

Trigger this skill for any security-flavored task:

- Threat triage or incident response
- MITRE ATT&CK technique mapping (T1078, T1055, …)
- CVE lookup or vulnerability analysis
- Exploit / payload / attacker-technique questions
- Red team / blue team / purple team exercises
- CTF challenge analysis
- Detection engineering (Sigma / Snort / YARA / KQL / SPL / LQL rules)
- Defensive control validation
- Compliance / hardening questions

**Prerequisite:** the RAG should be loaded with security-relevant content. The [`cybersecurity` preset](../presets/cybersecurity.yaml) is optimized for this — it ships 8 categories, 200+ routing keywords, and 69 query expansions covering `sqli` → `SQL injection`, `privesc` → `privilege escalation`, etc.

---

## What this skill commits to

For security tasks, the corpus is **primary**:

1. Search the corpus **before** hitting any external threat-intel MCP (`mcp__cti__*`, `mcp__virustotal__*`, `mcp__shodan__*`).
2. Use the query-expansion mappings (`sqli` → `SQL injection`) — they exist for a reason.
3. Cite the MITRE technique ID + the internal runbook link.
4. Only escalate to external threat-intel when the corpus does not cover the specific IOC / CVE / technique in enough depth.

---

## Steps

1. **Classify the intent.** Is it:
   - **Detection** — "we saw X in logs, is this bad?"
   - **Response** — "we confirmed compromise, what now?"
   - **Prevention** — "how do we harden against X?"
   - **Research** — "how does X actually work?"

2. **Extract the security signature** from the user's message:
   - MITRE ATT&CK ID (`T1078`, `T1078.001`, `TA0004`)
   - CVE ID (`CVE-2024-1234`)
   - Malware family (`Cobalt Strike`, `Mimikatz`, `LSASS dump`)
   - Attack technique term (`Kerberoasting`, `AS-REP Roasting`, `pass-the-hash`)
   - IOC (hash, domain, IP — treat as verbatim in the search)

3. **Search the corpus with the security preset in mind:**
   ```
   search_knowledge(query="<signature> <intent>", max_results=5, min_score=0.15)
   ```
   Example: `search_knowledge(query="LSASS dump credential access detection", max_results=5)`

4. **If the RAG has a routing preset,** the categories `redteam` / `blueteam` / `mitre` / `ctf` are likely present. Filter when specificity matters:
   ```
   search_knowledge(query="lateral movement WinRM", category="blueteam", max_results=5)
   ```

5. **Read the top hits and build the answer around them.** Typical outputs to include:

   | Section | Content |
   |---|---|
   | **Technique** | MITRE ATT&CK ID + name + tactic |
   | **Detection** | log signals, EDR queries (KQL / SPL / LQL), Sigma rules — from corpus |
   | **Response** | runbook steps from `blueteam` category or `def-*` skills |
   | **Attacker view** | tools / commands from `redteam` category (authorized engagements only) |
   | **References** | citations to corpus files + MITRE technique URL + CVE URL |

6. **Only escalate to external threat-intel MCPs when the corpus is thin on the specific IOC:**
   ```
   # Only if RAG returns generic-only hits, not IOC-specific:
   mcp__virustotal__vt_file_report(hash=...)
   mcp__cti__ti_check_ioc(ioc=..., ioc_type="hash")
   mcp__shodan__shodan_host(ip=...)
   ```

7. **After resolution**, chain to [`rag-index-decisions`](rag-index-decisions.md) if the incident produced a new detection rule, runbook entry, or lesson.

---

## Examples

**Example 1 — MITRE-mapped detection question**

User: *"We saw process creation of `rundll32.exe comsvcs.dll MiniDump 632 dump.dmp full` on a workstation. Is this bad?"*

Agent:
```
search_knowledge(query="rundll32 comsvcs MiniDump LSASS credential dump",
                 max_results=5, min_score=0.15)
→ 4 hits: mitre-T1003.001-lsass.md (0.94), redteam-cred-dumping.md (0.82),
          blueteam-lsass-detection.md (0.79), edr-tuning-lsass.md (0.71)

Reply: "Yes, high-confidence malicious [mitre-T1003.001-lsass.md].

**Technique:** MITRE ATT&CK T1003.001 (OS Credential Dumping: LSASS Memory)
via the built-in `comsvcs.dll` MiniDump helper. Legitimate use of this
command is extremely rare.

**Detection signals** [blueteam-lsass-detection.md]:
- Process: `rundll32.exe` with cmdline containing `comsvcs.dll` + `MiniDump`
- Parent process: often `cmd.exe` or `powershell.exe` after initial access
- File write: `.dmp` file with LSASS-typical size (~50-500 MB)

**Immediate response** [runbooks/incident-lsass-dump.md]:
1. Isolate the host (network quarantine, do not power off — preserve memory)
2. Capture the dump before attacker retrieves it
3. Rotate credentials of every user logged on to that host (last 24h)
4. Kerberos: krbtgt reset if attacker had time to Golden Ticket

**Related ATT&CK:** T1078 (Valid Accounts, likely next step),
T1550 (Alternate Auth Material)."
```

**Example 2 — CVE lookup**

User: *"Is CVE-2024-3400 relevant to our infra?"*

Agent:
```
search_knowledge(query="CVE-2024-3400", max_results=5) → 0 hits (specific CVE not indexed)
search_knowledge(query="PAN-OS GlobalProtect vulnerability", max_results=5) → 2 hits
  → asset-inventory-perimeter.md (0.68), vuln-response-playbook.md (0.55)

Reply: "The corpus does not have CVE-2024-3400 specifically indexed. Checking external
sources... [chains to mcp__cti__vuln_cve(cve_id='CVE-2024-3400')]

From external threat intel: CVE-2024-3400 is a critical PAN-OS GlobalProtect command
injection (CVSS 10.0), actively exploited since March 2024.

**Relevance to your infra** [asset-inventory-perimeter.md]:
- Your perimeter inventory lists 2 PAN-OS firewalls in scope.
- Response playbook: [vuln-response-playbook.md] — priority PATCH within 24h
  for internet-facing PAN-OS with GlobalProtect enabled.

**Next steps:**
1. Confirm PAN-OS version on those 2 firewalls
2. Check if GlobalProtect gateway feature is enabled (only vulnerable if yes)
3. If yes → apply hotfix per PAN advisory; if no → still upgrade but non-emergency
4. Hunt for IOCs from the campaign — chain to mcp__cti__ti_check_ioc"
```

**Example 3 — Red team, authorized engagement**

User: *"Doing an authorized internal pentest. Need Kerberoasting query and post-exploitation."*

Agent:
```
search_knowledge(query="Kerberoasting SPN request authorized pentest",
                 category="redteam", max_results=5)
→ 4 hits: kerberoasting-technique.md (0.91), rubeus-usage.md (0.83),
          post-exploit-cred-cracking.md (0.75), opsec-considerations.md (0.68)

Reply: "**MITRE ATT&CK T1558.003** (Steal or Forge Kerberos Tickets: Kerberoasting)
[kerberoasting-technique.md].

**Enumeration:**
```powershell
Get-ADUser -Filter {ServicePrincipalName -ne \"$null\"} -Properties ServicePrincipalName
# or Impacket:
GetUserSPNs.py DOMAIN/user:pass -dc-ip <DC>
```
[kerberoasting-technique.md#enumeration]

**Extraction with Rubeus:**
```
Rubeus.exe kerberoast /outfile:hashes.txt /nowrap
```
[rubeus-usage.md]

**Cracking:**
```
hashcat -m 13100 hashes.txt rockyou.txt --force
```
[post-exploit-cred-cracking.md]

**OPSEC** [opsec-considerations.md]: EDRs flag Rubeus / GetUserSPNs by
default — use only from authorized C2 or bastion. Prefer `Rubeus /aes` to
avoid RC4 detection."
```

---

## Edge cases

- **Corpus is not security-loaded** — `list_categories()` returns no security-flavored categories. Skill still works but degrades: no runbook citations, more escalation to external threat-intel MCPs.
- **Sensitive engagement content** — never quote client-specific IOCs from one engagement in another. Sanitize before responding.
- **Unauthorized attacker technique request** — if the user is asking how to attack a system without authorization, this skill does NOT help. Refuse per general policy — RAG-first is about *how to search*, not *what to search for*.
- **Live IOC that should not be cached** — if the response includes a live C2 IP or malware hash, do NOT `add_from_url` it into the RAG. Cite once and move on.

---

## Related skills

- **[`rag-check-first`](rag-check-first.md)** — the base pattern (this is `check-first` with security priorities).
- **[`rag-cite-sources`](rag-cite-sources.md)** — extra important in security (auditable trail matters).
- **[`rag-troubleshoot`](rag-troubleshoot.md)** — incident triage often overlaps with debugging.
- **[`rag-index-decisions`](rag-index-decisions.md)** — after handling an incident, index the postmortem.
