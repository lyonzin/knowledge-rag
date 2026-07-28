"""
╭─╴ KNOWLEDGE RAG · STOPWORDS ╶──────────────────────── v 4.6.0 ─╮
│                                                                │
│   Multilingual query stopword filtering for BM25 retrieval.    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Since   ·  v4.6.0
    └─ Date    ·  2026-07-27

Why this module exists
----------------------
BM25 scores every token in the query. Natural-language scaffolding — "how does
X work", "como funciona X", "wie funktioniert X" — contributes rank noise: the
scaffolding tokens either match nothing (wasted work) or match thousands of
chunks with near-zero IDF, diluting the signal of the terms the user actually
cares about. Worse, those tokens also feed query expansion, where a single
generic word can pull in a whole synonym group unrelated to the question.

Filtering runs on the *query* only. Indexed documents are never stripped — the
inverted index keeps full text so phrase-level evidence stays intact. Semantic
search also keeps the raw query, since sentence embeddings benefit from the
natural-language framing that hurts BM25.

Safety model
------------
Aggressive stopword removal is dangerous in a security/engineering corpus where
short function words collide with real technical identifiers ("os" → operating
system, "com" → COM object hijacking, "des" → DES cipher, "war" → Java WAR
archive, "die" → PHP ``die()``, "man" → man pages). Three layers guard that:

1. ``PROTECTED_TERMS`` — a hard allowlist subtracted from every language set at
   import time. A term in this set can never be filtered, regardless of which
   languages are active.
2. Acronym heuristic — a token written in all-caps with 2+ characters is treated
   as an identifier and kept ("OS", "DES", "WAR", "CVE-2024-1234"). Natural
   stopwords are written lowercase or sentence-case; acronyms are not.
3. Empty-result guard — if filtering would remove every token, the original
   query is returned untouched. A degraded query always beats no query.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "DEFAULT_LANGUAGES",
    "MULTILINGUAL_STOPWORDS",
    "PROTECTED_TERMS",
    "SUPPORTED_LANGUAGES",
    "filter_query_stopwords",
    "get_stopwords",
    "normalize_language_codes",
]


# ╭─╴ Protected terms ╶─────────────────────── v 4.6.0 ─╮
# │   Never filtered, whatever the active languages     │
# ╰─────────────────────────────────────────────────────╯

# fmt: off
PROTECTED_TERMS: frozenset[str] = frozenset({
    # HTTP verbs — searched verbatim in web-security material
    "get", "post", "put", "head", "patch", "delete", "options", "trace", "connect",
    # SQL / boolean operators — in this corpus the payload *is* the query
    "and", "or", "not", "null", "union", "select", "from", "where", "order", "group",
    # Technical identifiers that collide with function words in the supported
    # languages. Every entry below is a term users genuinely search for.
    "as",    # AS-REP roasting, autonomous system
    "com",   # COM object hijacking (T1546.015), .com
    "da",    # Domain Admin
    "der",   # DER certificate encoding
    "des",   # DES cipher
    "die",   # PHP die()
    "es",    # Elasticsearch, event sourcing
    "man",   # man pages
    "net",   # net.exe, .NET
    "os",    # OS command injection, operating system
    "war",   # Java WAR archive
    "bald",  # explicit carve-out: never treat as a German stopword
})
# fmt: on


# ╭─╴ Language sets ╶───────────────────────── v 4.6.0 ─╮
# │   Question words · auxiliaries · connectives        │
# ╰─────────────────────────────────────────────────────╯
#
# Scope is deliberately narrow: interrogatives, copulas/auxiliaries, articles
# and prepositions, plus the handful of generic "query verbs" that carry no
# retrieval signal ("work", "funciona", "funktioniert"). Content verbs and nouns
# are never listed — IDF already handles moderately common words, and every
# extra entry is another chance to delete a real query term.

# fmt: off
_RAW_STOPWORDS: dict[str, set[str]] = {
    "en": {
        # interrogatives
        "what", "whats", "what's", "how", "hows", "how's", "why", "when",
        "where", "who", "whos", "who's", "whom", "whose", "which",
        # copulas / auxiliaries / modals
        "is", "are", "am", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "has", "have", "had", "having",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
        # determiners / pronouns / prepositions
        "the", "a", "an", "this", "that", "these", "those", "there", "here",
        "it", "its", "i", "you", "we", "they", "he", "she",
        "my", "your", "our", "their",
        "of", "to", "in", "for", "with", "about", "on", "at", "by", "into",
        "over", "between",
        # generic query verbs
        "work", "works", "working", "mean", "means",
        "explain", "explains", "tell", "please",
    },
    "pt": {
        # interrogativas
        "como", "qual", "quais", "quando", "onde", "aonde", "quem",
        "porque", "porquê", "pq", "que", "oque",
        # cópulas / auxiliares
        "é", "eh", "sao", "são", "ser", "sendo", "esta", "está", "estao",
        "estão", "estar", "faz", "fazem", "fazer", "tem", "têm", "ter",
        "pode", "podem", "poder", "deve", "devem", "vai", "vao", "vão",
        "foi", "foram", "seria",
        # artigos / preposições / pronomes
        "o", "a", "os", "um", "uma", "uns", "umas",
        "de", "do", "dos", "das", "em", "no", "na", "nos", "nas",
        "ao", "aos", "à", "às",
        "para", "pra", "por", "pelo", "pela", "sem", "sobre", "entre",
        "e", "ou", "se", "isso", "isto", "esse", "essa", "este",
        "eu", "voce", "você", "eles", "elas",
        # verbos genéricos de pergunta
        "funciona", "funcionam", "funcionar", "significa", "significam",
        "explica", "explicar",
    },
    "es": {
        # interrogativos
        "qué", "que", "cómo", "como", "cuál", "cuáles", "cuándo", "cuando",
        "dónde", "donde", "quién", "quien", "porqué", "porque", "por",
        # cópulas / auxiliares
        "son", "ser", "está", "están", "estar", "hace", "hacen", "hacer",
        "tiene", "tienen", "tener", "puede", "pueden", "debe", "deben",
        "va", "fue", "sería",
        # artículos / preposiciones / pronombres
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "del", "en", "al", "a", "y", "o", "con", "sin", "sobre", "entre",
        "para", "esto", "eso", "este", "esta", "ese", "esa",
        # verbos genéricos de pregunta
        "funciona", "funcionan", "funcionar", "significa", "explica", "explicar",
    },
    "de": {
        # Fragewörter
        "was", "wie", "warum", "wieso", "weshalb", "wann", "wo", "wer",
        "wen", "wem", "wessen", "welche", "welcher", "welches",
        # Kopula / Hilfsverben / Modalverben
        "ist", "sind", "bin", "waren", "sein", "hat", "haben", "hatte",
        "kann", "können", "koennen", "soll", "sollen", "muss", "müssen",
        "wird", "werden", "würde",
        # Artikel / Präpositionen / Pronomen
        "das", "den", "dem", "ein", "eine", "einen", "einem", "einer", "eines",
        "und", "oder", "nicht", "mit", "von", "zu", "für", "fuer",
        "auf", "im", "am", "an", "bei", "aus", "über", "ueber",
        "ich", "du", "sie", "wir",
        # generische Frageverben
        "funktioniert", "funktionieren", "bedeutet", "erklärt", "erklaeren",
    },
    "fr": {
        # interrogatifs
        "que", "qu", "quoi", "comment", "pourquoi", "quand", "où", "ou",
        "qui", "quel", "quelle", "quels", "quelles",
        # copules / auxiliaires
        "est", "sont", "être", "etre", "ont", "avoir", "fait", "font",
        "faire", "peut", "peuvent", "doit", "doivent", "va", "était",
        # articles / prépositions / pronoms
        "le", "la", "les", "un", "une", "du", "de", "au", "aux",
        "et", "dans", "pour", "avec", "sans", "sur", "par", "en",
        "ne", "pas", "ce", "cette", "ces", "cela",
        # verbes génériques de question
        "fonctionne", "fonctionnent", "fonctionner", "signifie", "explique",
        "expliquer",
    },
    "it": {
        # interrogativi
        "che", "cosa", "come", "perché", "perche", "quando", "dove", "chi",
        "quale", "quali",
        # copule / ausiliari
        "è", "sono", "essere", "ha", "hanno", "avere", "fa", "fanno", "fare",
        "può", "puo", "possono", "deve", "devono", "va", "era",
        # articoli / preposizioni / pronomi
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "di", "della", "dei", "delle", "in", "nel", "nella",
        "a", "al", "alla", "dal", "per", "con", "su", "tra", "fra",
        "non", "questo", "questa", "quello",
        # verbi generici di domanda
        "funziona", "funzionano", "funzionare", "significa", "spiega", "spiegare",
    },
}
# fmt: on


MULTILINGUAL_STOPWORDS: dict[str, frozenset[str]] = {
    lang: frozenset(words - PROTECTED_TERMS) for lang, words in _RAW_STOPWORDS.items()
}
"""Per-language stopword sets, already stripped of :data:`PROTECTED_TERMS`."""

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(sorted(MULTILINGUAL_STOPWORDS))
"""Language codes accepted by :func:`filter_query_stopwords` (ISO 639-1)."""

DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "pt")
"""Languages enabled when the caller does not configure an explicit list."""


# Characters trimmed from token edges before the stopword lookup. Only the edges
# are touched, so "example.com", "use-after-free" and "CVE-2024-1234" survive.
_EDGE_PUNCTUATION = " \t\r\n?!.,;:()[]{}<>\"'`«»¿¡…"


# ╭─╴ Public API ╶──────────────────────────── v 4.6.0 ─╮
# │   normalize · get_stopwords · filter_query          │
# ╰─────────────────────────────────────────────────────╯


def normalize_language_codes(languages: list[str] | None) -> tuple[str, ...]:
    """
    Normalize caller-supplied language codes to supported ISO 639-1 codes.

    Accepts regional variants (``pt-BR``, ``en_US``) and arbitrary casing.
    Unrecognized codes are dropped silently rather than raising — a typo in
    ``config.yaml`` must never take search down.

    Three inputs are deliberately distinguished:

    - ``None`` → every supported language (the caller expressed no preference).
    - ``[]`` → no language at all, i.e. filtering is switched off. An explicit
      empty list is an explicit opt-out.
    - a non-empty list where nothing is recognized → :data:`DEFAULT_LANGUAGES`,
      because a typo should degrade to the shipped default rather than silently
      disable a feature the operator asked for.

    Args:
        languages: Requested language codes, ``None`` for every supported
            language, or ``[]`` to disable filtering.

    Returns:
        tuple[str, ...]: Deduplicated, sorted, supported codes.

    Example:
        >>> normalize_language_codes(["pt-BR", "EN_us", "klingon"])
        ('en', 'pt')
        >>> normalize_language_codes([])
        ()
    """
    if languages is None:
        return SUPPORTED_LANGUAGES
    if not languages:
        return ()  # explicit opt-out

    resolved: set[str] = set()
    for raw in languages:
        code = str(raw).strip().lower().replace("_", "-").split("-", 1)[0]
        if code in MULTILINGUAL_STOPWORDS:
            resolved.add(code)

    if not resolved:
        return DEFAULT_LANGUAGES
    return tuple(sorted(resolved))


@lru_cache(maxsize=32)
def get_stopwords(languages: tuple[str, ...]) -> frozenset[str]:
    """
    Build (and cache) the union of stopword sets for the given languages.

    Args:
        languages: Already-normalized language codes. Must be a tuple so the
            result can be memoized.

    Returns:
        frozenset[str]: Every stopword active for those languages.

    Example:
        >>> "how" in get_stopwords(("en",))
        True
        >>> "os" in get_stopwords(("pt",))  # protected identifier
        False
    """
    union: set[str] = set()
    for code in languages:
        union |= MULTILINGUAL_STOPWORDS.get(code, frozenset())
    return frozenset(union)


def _is_acronym(token: str) -> bool:
    """
    Report whether a token looks like an all-caps technical identifier.

    Args:
        token: Punctuation-trimmed token in its original casing.

    Returns:
        bool: True for 2+ character tokens with no lowercase letters and at
        least one cased character ("OS", "DES", "CVE-2024-1234").
    """
    return len(token) >= 2 and token.isupper()


def filter_query_stopwords(query: str, languages: list[str] | None = None) -> str:
    """
    Strip natural-language scaffolding from a search query.

    Removes interrogatives, auxiliaries, articles and generic query verbs so
    BM25 scores only the terms carrying retrieval signal. Token order and the
    original spelling of surviving tokens are preserved, so downstream bigram
    lookups in query expansion keep working.

    Never removes a token listed in :data:`PROTECTED_TERMS` or one that reads as
    an all-caps acronym. If filtering would empty the query, the original string
    is returned unchanged — a noisy query outranks an empty one.

    Args:
        query: Raw user query. May be empty or whitespace-only.
        languages: ISO 639-1 codes to filter against (default: ``None``, which
            applies every supported language). Regional variants such as
            ``pt-BR`` are accepted. Pass ``[]`` to disable filtering.

    Returns:
        str: The filtered query, or the original ``query`` when filtering
        produced nothing (including empty/whitespace input).

    Example:
        >>> filter_query_stopwords("how does SSRF work", ["en"])
        'SSRF'
        >>> filter_query_stopwords("como funciona ldap kerberoast", ["pt"])
        'ldap kerberoast'
        >>> filter_query_stopwords("how what why", ["en"])  # nothing survives
        'how what why'
    """
    if not query or not query.strip():
        return query

    stopwords = get_stopwords(normalize_language_codes(languages))
    if not stopwords:
        return query

    kept: list[str] = []
    for raw_token in query.split():
        core = raw_token.strip(_EDGE_PUNCTUATION)
        if not core:
            kept.append(raw_token)
            continue
        lowered = core.lower()
        if lowered in PROTECTED_TERMS or _is_acronym(core) or lowered not in stopwords:
            kept.append(raw_token)

    if not kept:
        return query
    return " ".join(kept)
