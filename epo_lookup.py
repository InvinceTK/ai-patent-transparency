import requests
import base64
import xml.etree.ElementTree as ET
import json
import time

PUBLIC_KEY = "Bn5NAjAOyJ6q1xF7CvXN0d2aUl03JAHirEfT1uSRvOLAXeOb"
SECRET_KEY = "QCiL59EoeObZVtDwktdUPWIsibqC5AAzrft2RG3lREBm69UHYD8rkJeJPrStKseY"
NS = "{http://www.epo.org/exchange}"
NS_OPS = "{http://ops.epo.org}"
NS_FT = "{http://www.epo.org/fulltext}"


# ──────────────────────────────────────────────
# CQL QUERY BUILDER
# ──────────────────────────────────────────────

def build_cql_query(
    keywords: list[str],
    operator: str = "or",
    search_fields: list[str] = None,
    applicant: str = "BOEING CO",
) -> str:
    """
    Build a CQL query string.

    Args:
        keywords:      List of words / phrases to search for.
        operator:      Boolean operator joining keywords: 'or' | 'and'.
        search_fields: Fields to search in. Supported values:
                         'ti'  – title
                         'ab'  – abstract
                         'txt' – all text fields (title + abstract + description + claims)
                       Defaults to ['ti', 'ab', 'txt'] (broadest coverage without duplicating).
        applicant:     Applicant name to restrict results to (EPO epodoc format).
                       Pass None to skip the applicant filter.

    Returns:
        A CQL query string ready to be URL-encoded and sent to OPS.

    Examples
    --------
    >>> build_cql_query(["wingman", "uncrewed", "teaming"])
    '(ti=wingman or ab=wingman or txt=wingman) and (ti=uncrewed or ...) and pa="BOEING CO"'
    """
    if search_fields is None:
        search_fields = ["ti", "ab", "txt"]

    op = f" {operator.strip()} "

    # For each keyword build a clause searching across all requested fields
    keyword_clauses = []
    for kw in keywords:
        # Quote multi-word phrases; single words don't need quotes
        term = f'"{kw}"' if " " in kw else kw
        field_clause = op.join(f"{field}={term}" for field in search_fields)
        keyword_clauses.append(f"({field_clause})")

    # Join keyword groups with AND so every keyword must appear somewhere
    combined = " or ".join(keyword_clauses)

    if applicant and combined:
        combined = f"({combined}) and pa={applicant}"
    elif applicant:
        combined = f"pa={applicant}"

    return combined


# ──────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────

def get_access_token() -> str:
    credentials = f"{PUBLIC_KEY}:{SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode()).decode()
    response = requests.post(
        "https://ops.epo.org/3.2/auth/accesstoken",
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data="grant_type=client_credentials",
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ──────────────────────────────────────────────
# Fulltext is only available for these authorities (per EPO docs).
# Order matters — for each patent family the first match wins.
FULLTEXT_AUTHORITIES = ["EP", "WO", "AT", "BE", "BG", "CA", "CH", "CY", "CZ", "DK",
                        "EE", "ES", "FR", "GB", "GR", "HR", "IE", "IT", "LT", "LU",
                        "MC", "MD", "ME", "NO", "PL", "PT", "RO", "RS", "SE", "SK"]

# ──────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────

def _fetch_families_page(cql_query: str, token: str, start: int, end: int) -> tuple[int, dict[str, list[str]]]:
    """
    Fetch one page of biblio search results.
    Returns (total_result_count, families) where families maps family_id -> list[epodoc_id].
    """
    url = "https://ops.epo.org/rest-services/published-data/search/biblio"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "X-OPS-Range": f"{start}-{end}",
        },
        params={"q": cql_query},
    )
    if response.status_code == 404:
        return 0, {}
    response.raise_for_status()

    tree = ET.fromstring(response.text)
    total_el = tree.find(f".//{NS_OPS}biblio-search")
    total = int(total_el.attrib.get("total-result-count", 0)) if total_el is not None else 0

    # Each exchange-document is one jurisdiction filing; family-id groups them.
    families: dict[str, list[str]] = {}
    for doc in tree.findall(f".//{NS}exchange-document"):
        family_id = doc.attrib.get("family-id", "unknown")
        doc_id_el = doc.find(f".//{NS}document-id[@document-id-type='epodoc']/{NS}doc-number")
        if doc_id_el is None or not doc_id_el.text:
            continue
        pid = doc_id_el.text.strip()
        families.setdefault(family_id, []).append(pid)

    return total, families


def _pick_by_jurisdiction(all_families: dict[str, list[str]]) -> list[str]:
    """For each family pick the highest-priority jurisdiction from FULLTEXT_AUTHORITIES."""
    chosen_ids = []
    for family_id, pids in all_families.items():
        by_country: dict[str, str] = {}
        for pid in pids:
            country = "".join(c for c in pid[:4] if c.isalpha())[:2].upper()
            by_country[country] = pid

        for country in FULLTEXT_AUTHORITIES:
            if country in by_country:
                chosen_ids.append(by_country[country])
                break
        else:
            print(f"  Skipping family {family_id} — no supported jurisdiction ({list(by_country.keys())})")

    return chosen_ids


def search_patents(cql_query: str, token: str, start: int = 1, end: int = 25) -> list[str]:
    """
    Fetch a single page of results and return one epodoc ID per family.
    Use search_all_patents() to paginate through all available results.
    """
    total, families = _fetch_families_page(cql_query, token, start, end)
    print(f"  Total results available: {total} (fetching {start}-{end})")
    chosen_ids = _pick_by_jurisdiction(families)
    print(f"  Unique families: {len(families)} -> kept {len(chosen_ids)} after jurisdiction filter")
    return chosen_ids


def search_all_patents(cql_query: str, token: str, page_size: int = 100, max_results: int = 2000) -> list[str]:
    """
    Paginate through all search results and return one epodoc patent ID per family.

    Fetches up to max_results results in pages of page_size (EPO OPS hard limit: 100/page,
    2000 total). Families are merged across pages before the jurisdiction filter is applied,
    so each invention is represented by exactly one ID regardless of which page it appears on.
    """
    page_size = min(page_size, 100)  # EPO OPS hard cap

    all_families: dict[str, list[str]] = {}

    # First page also tells us the total count
    start, end = 1, min(page_size, max_results)
    total, families = _fetch_families_page(cql_query, token, start, end)
    cap = min(total, max_results)
    print(f"  Total results available: {total} (fetching up to {cap})")

    for fid, pids in families.items():
        all_families.setdefault(fid, []).extend(pids)

    fetched = end
    while fetched < cap:
        start = fetched + 1
        end = min(fetched + page_size, cap)
        time.sleep(0.2)  # polite rate-limiting between pages
        _, families = _fetch_families_page(cql_query, token, start, end)
        for fid, pids in families.items():
            all_families.setdefault(fid, []).extend(pids)
        fetched = end
        print(f"  Fetched {fetched}/{cap}")

    chosen_ids = _pick_by_jurisdiction(all_families)
    print(f"  Unique families: {len(all_families)} -> kept {len(chosen_ids)} after jurisdiction filter")
    return chosen_ids


# ──────────────────────────────────────────────
# INDIVIDUAL PATENT DATA RETRIEVAL
# ──────────────────────────────────────────────

def get_biblio(patent_id: str, token: str) -> dict:
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/biblio"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return {}
    tree = ET.fromstring(response.text)
    title_el = tree.find(f".//{NS}invention-title")
    country_el = tree.find(f".//{NS}country")
    date_el = tree.find(f".//{NS}date")
    kind_el = tree.find(f".//{NS}kind")
    doc_number_el = tree.find(f".//{NS}doc-number")
    return {
        "title": title_el.text if title_el is not None else None,
        "country": country_el.text if country_el is not None else None,
        "date": date_el.text if date_el is not None else None,
        "kind": kind_el.text if kind_el is not None else None,
        "doc_number": doc_number_el.text if doc_number_el is not None else None,
    }


def get_abstract(patent_id: str, token: str) -> str | None:
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/abstract"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return None
    tree = ET.fromstring(response.text)
    abstract_el = tree.find(f".//{NS}abstract/{NS}p")
    return abstract_el.text if abstract_el is not None else None


def get_description(patent_id: str, token: str) -> str | None:
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/description"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return None
    tree = ET.fromstring(response.text)
    paragraphs = [p.text for p in tree.findall(f".//{NS_FT}description/{NS_FT}p") if p.text]
    return " ".join(paragraphs) if paragraphs else None


def get_claims(patent_id: str, token: str) -> list[str] | None:
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/claims"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return None
    tree = ET.fromstring(response.text)
    claims = [c.text for c in tree.findall(f".//{NS_FT}claim-text") if c.text]
    return claims if claims else None


def get_patent_info(patent_id: str, token: str) -> dict:
    """Fetch all available data for a single patent ID."""
    biblio = get_biblio(patent_id, token)
    return {
        "patent_id": patent_id,
        **biblio,
        "abstract": get_abstract(patent_id, token),
        "description": get_description(patent_id, token),
        "claims": get_claims(patent_id, token),
    }


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

def print_patent_info(info: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Patent ID:   {info.get('patent_id')}")
    print(f"Title:       {info.get('title')}")
    print(f"Country:     {info.get('country')}")
    print(f"Date:        {info.get('date')}")
    print(f"Kind:        {info.get('kind')}")
    print(f"\nAbstract:\n{info.get('abstract')}")
    desc = info.get("description") or ""
    print(f"\nDescription (first 500 chars):\n{desc[:500]}")
    claims = info.get("claims") or []
    print(f"\nClaims ({len(claims)} total):")
    for i, claim in enumerate(claims[:3], 1):   # show first 3 only
        print(f"  [{i}] {claim[:200]}")
    if len(claims) > 3:
        print(f"  ... and {len(claims) - 3} more.")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("EPO Patent Search — CQL / Boeing CO\n")

    # ── Configure your search here ─────────────────────────────────────────
    # Keywords from the image (feel free to edit or extend this list)
    KEYWORDS = []

    # 'or'  -> patent must contain AT LEAST ONE keyword  (recommended for broad search)
    # 'and' -> patent must contain ALL keywords          (very restrictive -- use 2-3 keywords max)
    KEYWORD_OPERATOR = "or"

    # 'txt' covers title + abstract + description + claims in one index.
    # Using ti/ab/txt together triples query size and causes 413 errors.
    # Stick to ['txt'] unless you specifically need title-only or abstract-only.
    SEARCH_FIELDS = ["txt"]

    # Restrict to Boeing only (set to None to search all applicants)
    APPLICANT = "optum"

    # How many results to fetch total (max 2000 per EPO OPS)
    MAX_RESULTS = 2000

    # Whether to fetch full details (abstract, description, claims) for each hit
    FETCH_FULL_DETAILS = True
    # ───────────────────────────────────────────────────────────────────────

    token = get_access_token()
    print("Access token received.\n")

    cql = build_cql_query(
        keywords=KEYWORDS,
        operator=KEYWORD_OPERATOR,
        search_fields=SEARCH_FIELDS,
        applicant=APPLICANT,
    )
    print(f"CQL Query:\n  {cql}\n")
    print(f"Query length: {len(cql)} chars")

    # EPO OPS rejects queries over ~2000 chars with a 413 error.
    # If you hit this, use SEARCH_FIELDS = ['txt'] only and/or reduce keywords.
    if len(cql) > 1800:
        raise ValueError(
            f"CQL query is {len(cql)} chars — too long for EPO OPS (limit ~2000).\n"
            "Fix: set SEARCH_FIELDS = ['txt'] and/or reduce the number of keywords."
        )

    patent_ids = search_all_patents(cql, token, max_results=MAX_RESULTS)
    print(f"\nReturned {len(patent_ids)} unique patent IDs.\n")

    results = []
    for i, pid in enumerate(patent_ids, 1):
        print(f"[{i}/{len(patent_ids)}] Fetching: {pid}")
        if FETCH_FULL_DETAILS:
            info = get_patent_info(pid, token)
            time.sleep(0.1)   # polite rate-limiting
        else:
            info = {"patent_id": pid}
        results.append(info)
        # print_patent_info(info)

    output_file = "optum.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} patents to {output_file}")

# patent ID looks like: EP1000000.A1

# Note, Currently full texts (description and/or claims) are available for the following
# authorities: EP, WO, AT, BE, BG, CA, CH, CY, CZ, DK, EE, ES, FR, GB, GR, HR, IE, IT,
# LT, LU, MC, MD, ME, NO, PL, PT, RO, RS, SE and SK. | NOT US,BR, JP



# Explanation of how it works

#  One flow through the pipeline:                                                                                                          
                                                                                                                                          
#   1. Build CQL query (build_cql_query) — turns your keywords + applicant into a CQL string like (txt=wingman or txt=teaming) and          
#   pa="BOEING CO"                                                                                                                          
#   2. Search biblio (search_all_patents → _fetch_families_page) — sends that CQL to /published-data/search/biblio in pages of 100. Each    
#   response is a list of exchange-document XML elements — one per jurisdiction filing (e.g. the same invention filed as EP, US, WO all come
#    back as separate documents, each tagged with a family-id).                                                                             
#   3. Group by family — all documents sharing a family-id are bucketed together. One family = one invention, multiple country filings.     
#   4. Filter families (_pick_by_jurisdiction) — for each family, walk the FULLTEXT_AUTHORITIES priority list (EP first, then WO, then
#   rest). Pick the first jurisdiction that has a filing. Families with no supported jurisdiction are dropped. Result: one epodoc ID per
#   invention (e.g. EP3456789A1).
#   5. Fetch full text (get_patent_info) — for each chosen ID, makes 3 separate API calls:
#     - /biblio → title, country, date, kind
#     - /abstract → abstract paragraph
#     - /description + /claims → full text (only works for the whitelisted jurisdictions from step 4 — that's why the filter matters)
#   6. Save — all results dumped to boeing_patents.json.

#   The reason EP/WO are preferred in step 4: the EPO fulltext API only serves description+claims for those specific authorities. A US
#   filing of the same patent would return nothing from steps 5's description/claims calls.