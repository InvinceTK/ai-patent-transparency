import requests
import base64
import xml.etree.ElementTree as ET
import json 

PUBLIC_KEY = "Bn5NAjAOyJ6q1xF7CvXN0d2aUl03JAHirEfT1uSRvOLAXeOb"
SECRET_KEY = "QCiL59EoeObZVtDwktdUPWIsibqC5AAzrft2RG3lREBm69UHYD8rkJeJPrStKseY"
NS = "{http://www.epo.org/exchange}"
NS_FT = "{http://www.epo.org/fulltext}"


def get_access_token():
    credentials = f"{PUBLIC_KEY}:{SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
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


def get_biblio(patent_id, token):
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


def get_abstract(patent_id, token):
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/abstract"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return None
    tree = ET.fromstring(response.text)
    abstract_el = tree.find(f".//{NS}abstract/{NS}p")
    return abstract_el.text if abstract_el is not None else None


def get_description(patent_id, token):
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/description"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        print(f"Description 404 body: {response.text[:500]}")
        return None
    tree = ET.fromstring(response.text)
    paragraphs = [p.text for p in tree.findall(f".//{NS_FT}description/{NS_FT}p") if p.text]
    return " ".join(paragraphs) if paragraphs else None


def get_claims(patent_id, token):
    url = f"https://ops.epo.org/rest-services/published-data/publication/epodoc/{patent_id}/claims"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    print(f"Response status is {response}")
    if response.status_code != 200:
        return None
    tree = ET.fromstring(response.text)
    claims = [c.text for c in tree.findall(f".//{NS_FT}claim-text") if c.text]
    print(f"Claims are: {claims}")
    return claims if claims else None


def get_patent_info(patent_id):
    token = get_access_token()
    print("Access token recieved: ",token)
    biblio = get_biblio(patent_id, token)
    return {
        "patent_id": patent_id,
        **biblio,
        "abstract": get_abstract(patent_id, token),
        "description": get_description(patent_id, token),
        "claims": get_claims(patent_id, token),
    }

def print_patent_info(info):
    """
    Description : Prints out what patent looks like
    Input : Patent object
    """
    print(f"\nTitle:       {info.get('title')}")
    print(f"Country:     {info.get('country')}")
    print(f"Date:        {info.get('date')}")
    print(f"Kind:        {info.get('kind')}")
    print(f"\nAbstract:\n{info.get('abstract')}")
    print(f"\nDescription (first 500 chars):\n{(info.get('description') or '')[:500]}")
    claims = info.get("claims") or []
    print(f"\nClaims ({len(claims)} total):")
    for i, claim in enumerate(claims, 1):
        print(f"  [{i}] {claim[:200]}")

if __name__ == "__main__":
    """
    Description : Gets a patent from EPO api and saves it to sample.json
    Output : sample.json

    Patent object:
        patent_id,
        bilio
        abstract
        description
        claims
    """

    patent_id = input("Enter patent ID (e.g. EP1000000.A1): ").strip()
    patent = get_patent_info(patent_id)
    print_patent_info(patent)

    json_file = []
    json_file.append(patent)

    with open("sample.json", "w") as f:
        json.dump(json_file, f, indent=2)
    



# patent ID looks like: EP1000000.A1

# Note, Currently full texts (description and/or claims) are available for the following
# authorities: EP, WO, AT, BE, BG, CA, CH, CY, CZ, DK, EE, ES, FR, GB, GR, HR, IE, IT,
# LT, LU, MC, MD, ME, NO, PL, PT, RO, RS, SE and SK. | NOT US

