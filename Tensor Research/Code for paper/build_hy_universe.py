"""Curate the HY/crossover CDS universe link table (Markit -> CRSP/Compustat).

Inputs (already pulled, pre_prediction_cache/):
  markit_us_5y_snrfor_all.csv   970 US 5Y USD SNRFOR entities, median spread
  crsp_stocknames_2021plus.csv  CRSP name history valid in the CDS window
  ccm_linktable_full.csv        CCM gvkey<->permno (LC/LU, P/C)

Selection: median 5Y spread >= 150bp over 2021-07..2026-07 ("material").

Match rules (BR/Burlington lesson: ticker alone is NOT sufficient):
  1. Normalize names (abbreviation expansion, stop-word removal).
  2. Candidates = CRSP ticker match + best fuzzy name match.
  3. ACCEPT iff significant name tokens of one side are contained in the
     other (>=1 token, each len>=5, or >=2 tokens) — the containment
     direction CRSP-in-Markit deliberately captures financing/operating
     subsidiaries of listed parents (Ford Motor Credit -> F), which are the
     liquid reference entities for those capital structures —
     or SequenceMatcher ratio >= 0.92.
  4. Explicit whitelist/blacklist for names the rules cannot decide.
  5. Dedup: one redcode per gvkey (most CDS observations wins), one gvkey
     per redcode.

Output: pre_prediction_cache/hy_universe_link.csv
        (gvkey, permno, redcode, mticker, shortname, comnam, med_bp, n_obs,
         how, accepted-rule) + hy_universe_audit.csv with rejects.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "pre_prediction_cache"
MIN_BP = 150.0

ABBREV = {
    "AMERN": "AMERICAN", "INTL": "INTERNATIONAL", "GEN": "GENERAL",
    "MTR": "MOTOR", "MTRS": "MOTORS", "CENTY": "CENTURY", "ALUM": "ALUMINUM",
    "COML": "COMMERCIAL", "CMNTY": "COMMUNITY", "SYS": "SYSTEMS",
    "HLTH": "HEALTH", "SVCS": "SERVICES", "SVC": "SERVICE",
    "RES": "RESOURCES", "PDTS": "PRODUCTS", "PRODS": "PRODUCTS",
    "PROD": "PRODUCTS", "STL": "STEEL", "UTD": "UNITED", "FINL": "FINANCIAL",
    "RLTY": "REALTY", "INVT": "INVESTMENT", "PAC": "PACIFIC",
    "STHN": "SOUTHERN", "STHWESTN": "SOUTHWESTERN", "ENGY": "ENERGY",
    "ENTMT": "ENTERTAINMENT", "COMMS": "COMMUNICATIONS",
    "COMWLTH": "COMMONWEALTH", "TECHS": "TECHNOLOGIES", "TECH": "TECHNOLOGY",
    "MGMT": "MANAGEMENT", "MFG": "MANUFACTURING", "PPTYS": "PROPERTIES",
    "PPTY": "PROPERTY", "BLDG": "BUILDING", "BLDRS": "BUILDERS",
    "BRDS": "BRANDS", "AIRLS": "AIRLINES", "HLDG": "HOLDING",
    "PKG": "PACKAGING", "CHEM": "CHEMICAL", "PWR": "POWER",
    "NAT": "NATURAL", "STRS": "STORES", "CR": "CREDIT", "FIN": "FINANCE",
    "FING": "FINANCING", "FDG": "FUNDING", "REINS": "REINSURANCE",
    "ENTPERS": "ENTERPRISES", "WKS": "WORKS", "GLOB": "GLOBAL",
    "EXPL": "EXPLORATION", "INDS": "INDUSTRIES", "RENT": "RENTAL",
    "OPER": "OPERATING", "PUBG": "PUBLISHING",
}
STOP = {"INC", "CORP", "CO", "LTD", "LLC", "LP", "PLC", "HOLDINGS", "HLDGS",
        "GROUP", "GRP", "COS", "COMPANY", "COMPANIES", "THE", "NEW", "DEL",
        "CL", "CORPORATION", "INCORPORATED", "PARTNERSHIP", "PARTNERS",
        "TRUST", "TR", "US", "USA", "GP", "SA", "NV", "DE"}

# shortname (upper, raw) -> CRSP ticker, decided by eye where rules fail.
WHITELIST = {
    "$ GEN CORP": "DG",                      # Dollar General
    "BATH BODY WKS INC": "BBWI",
    "SIRIUS XM RADIO INC": "SIRI",
    "SIRIUS XM RADIO LLC": "SIRI",
    "SABRE HLDGS CORP": "SABR",
    "ISTAR INC": "STAR",
    "AVIENT CORP": "AVNT",
    "EVEREST REINS HLDGS INC": "EG",         # Everest Group (was RE)
    "NUANCE COMMS INC": "NUAN",
    "NABORS INDS INC": "NBR",
    "BLUE OWL CAP CORP": "OBDC",
    "ONEMAIN FIN CORP": "OMF",
    "DISCOVERY COMMS LLC": "WBD",            # Discovery -> Warner Bros Disc
    "LUMEN TECH INC": "LUMN",
    "FRONTIER COMMS HLDGS LLC": "FYBR",
    "HOUGHTON MIFFLIN HARCOURT PUBLISHING CO": "HMHC",
    "CLEVELANDCLIFFS INC": "CLF",
    "IHEARTCOMMUNICATIONS INC": "IHRT",
    "K HOVNANIAN ENTPERS INC": "HOV",
    "BLDRS FIRSTSOURCE INC": "BLDR",
    "GRAY TELEVISION INC": "GTN",
    "GRAY MEDIA INC": "GTN",
    "UNIVISION COMMS INC": None,             # private since 2007 delist
    "CHARTER COMMS OPER LLC": "CHTR",
    "CCO HLDGS LLC": "CHTR",                 # Charter capital structure
    "FORD MTR CR CO LLC": "F",
    "GEN MTRS FINL CO INC": "GM",
    "AVIS BUDGET CAR RENT LLC": "CAR",
    "MPT OPER PARTNERSHIP LP": "MPW",        # Medical Properties Trust opco
    "VORNADO RLTY LP": "VNO",
    "HIGHWOODS RLTY LP": "HIW",
    "BOSTON PPTYS LTD PARTNERSHIP": "BXP",
    "BRANDYWINE OPER PARTNERSHIP LP": "BDN",
    "EQM MIDSTREAM PARTNERS LP": "ETRN",     # Equitrans (pre-EQT merger)
    "TALEN ENERGY SUPPLY LLC": "TLN",
    "VICI NT CO INC": "VICI",
    "VICI PPTYS LP": "VICI",
    "MACYS RETAIL HLDGS LLC": "M",
    "MACY S RETAIL HLDGS INC": "M",
    "WYNN LAS VEGAS LLC": "WYNN",
    "DISH DBS CORP": "DISH",
    "PARAMOUNT GLOB": "PARA",
    "TENNECO LLC": "TEN",
    "TENNECO INC": "TEN",
    "SEARS HLDGS CORP": None,                # dead, no live equity
    "CABLEVISION SYS CORP": None,            # private (Altice 2016)
    "AMERN GREETINGS CORP": None,            # private 2013
}
BLACKLIST_SHORT = {
    # fuzzy hits that are wrong and have no valid public parent
    "HILCORP ENERGY I LP", "ARCHSTONE", "TRAVELPORT LLC",
    "LIBERTY INTERACTIVE LLC", "GRAPHIC PACKAGING INTL LLC",
    "JOHN HANCOCK LIFE INS CO USA", "STANDARD INDS INC",
    "ALLIANT HLDGS INTERLLC", "LIFEPOINT HEALTH", "REWORLD HLDG CORP",
    "CATALENT PHARMA SOLUTIONS INC", "HEALTH NET INC", "JARDEN LLC",
    "WILLIAM LYON HOMES", "DAUCH CORP", "OFFICE DEPOT INC", "1ST INDL LP",
    "FREEPORT MINERALS CORP", "ION MEDIA NETWORKS INC",
    "DIAMOND SPORTS GROUP LLC", "BUCKEYE PARTNERS L P", "GEN CABLE CORP",
    "VALSPAR CORP", "NORTEK INC", "FELCOR LODGING LP",
    "MOMENTIVE PERFORMANCE MATLS INC", "CAESARS ENT INC",
    "COOPER INDS LLC", "CONVERGYS CORP", "WARNERMEDIA HLDGS INC",
    "TERMINIX CO LLC", "FREEDOM MTG HLDGS LLC", "LEVEL 3 TELECOM LLC",
    "COLONIAL RLTY LP", "MGM GROWTH PPTYS OPER PARTNERSHIP LP",
    "RESIDENTIAL CAP LLC", "VISTRA OPERATIONS CO LLC", "RESOLUTE FP US INC",
    "APOLLO DEBT SOLUTIONS BDC", "ALLEGHENY ENGY SUPP CO LLC",
    "MEDIACOM LLC", "ADT SEC CORP", "STATER BROS HLDGS INC", "STAPLES INC",
    "BEMIS CO INC", "NFE FING LLC", "SAKS INC", "RENTPATH LLC",
    "INFOR US INC", "HERTZ CORP", "USG CORP", "KATE SPADE CO",
    "HERCULES LLC", "ARCELORMITTAL USA LLC", "WEA FIN LLC",
    "WR BERKLEY CORP", "DOLE FOOD CO INC", "POST APT HOMES LP",
    "EMBARQ CORP", "FIRST-CITIZENS BK & TR CO", "LEVEL 3 FING INC",
    "STHN CA EDISON CO", "COREWEAVE INC", "SMITHFIELD FOODS INC",
    "BURLINGTON COAT FACTORY WHSE CORP", "QWEST CORP",
    "GT ATLANTIC & PACIFIC TEA CO INC", "LIBERTY MUT INS CO",
    "LONG IS PWR AUTH", "VENTURE GLOB LNG INC",
    "XPLR INFRASTRUCTURE OPER PARTNERS LP", "CHROMALLOY CORP",
    "NATIONWIDE MUT INS CO", "TANGER PPTYS LTD PARTNERSHIP",
    "GENWORTH HLDGS INC", "XPO CNW INC", "UN CARBIDE CORP",
    "OFFICEMAX INCORPORATED", "MBIA INS CORP", "MICHAELS STORES INC",
    "BOMBARDIER CAP INC", "VWR FDG INC", "MANDALAY RESORT GP",
    "CSC HLDGS LLC", "SUPERVALU INC", "JANUS CAP GP INC",
    "QWEST CAP FDG INC", "NISSAN MTR ACCEP CO LLC", "NEWFIELD EXPL CO",
    "DPL INC", "PLY GEM INDS INC", "BRIGGS & STRATTON CORP",
    "QWEST COMMS INTL INC", "BLACKSTONE PRIVATE CR FD", "ASURION LLC",
    "SALLY HLDGS LLC", "CARESTREAM HEALTH INC", "APRIA HEALTHCARE GP INC",
    "EDISON MISSION ENGY", "COOPER TIRE RUBR CO LLC",
    "DISCOVERY GLOB HLDGS INC", "GLOBAL MARINE INC", "BE AEROSPACE INC",
    "NOBLE ENGY INC", "NEXTEL COMMS INC", "NAVISTAR INTL CORP",
    "VALASSIS COMMS INC", "COMMSCOPE INC", "LEXMARK INTL INC",
    "HEXION INC", "CENTEX LLC", "GENON ENERGY HOLDINGS LLC",
    "CONDUENT BUSINESS SERVICES LLC", "NV ENERGY INC", "CBI GROUP INC",
    "PRIDE INTL LLC", "CLOUD SOFTWARE GROUP INC", "NAVISTAR FINL CORP",
    "AVON PRODS INC", "KEY ENGY SVCS INC", "CALPINE CORP",
    "INTERVAL ACQUISITION CORP", "ATI INC", "EQT CORP",
    "NATL RETAIL PPTYS INC", "WENDYS INTL LLC",
}


def norm_tokens(s: str) -> list[str]:
    toks = re.sub(r"[^A-Za-z ]", " ", str(s).upper()).split()
    return [ABBREV.get(t, t) for t in toks if ABBREV.get(t, t) not in STOP]


def contained(a: list[str], b: list[str]) -> bool:
    if not a or not b:
        return False
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    if not set(small) <= set(big):
        return False
    return (len(small) >= 2) or (len(small[0]) >= 5)


def main() -> int:
    mk = pd.read_csv(CACHE / "markit_us_5y_snrfor_all.csv")
    mk = mk[mk["med_bp"] >= MIN_BP].copy()
    nm = pd.read_csv(CACHE / "crsp_stocknames_2021plus.csv")
    lk = pd.read_csv(CACHE / "ccm_linktable_full.csv", dtype={"gvkey": str})

    crsp = nm[["permno", "ticker", "comnam"]].dropna().drop_duplicates().copy()
    crsp["toks"] = crsp["comnam"].map(norm_tokens)
    crsp["key"] = crsp["toks"].str.join(" ")

    rows = []
    for _, m in mk.iterrows():
        short_u = re.sub(r"\s+", " ", str(m["shortname"]).upper()).strip()
        mtoks = norm_tokens(m["shortname"])
        mkey = " ".join(mtoks)

        wl = WHITELIST.get(short_u, "ABSENT")
        if short_u in BLACKLIST_SHORT or wl is None:
            rows.append({**m, "decision": "reject",
                         "rule": "blacklist/whitelist-none"})
            continue

        cand = pd.DataFrame()
        if wl != "ABSENT":
            cand = crsp[crsp["ticker"] == wl]
            rule = "whitelist"
        if cand.empty:
            tick = crsp[crsp["ticker"] == m["ticker"]]
            scores = crsp["key"].map(
                lambda k: SequenceMatcher(None, mkey, k).ratio())
            best_i = scores.idxmax()
            options = []
            for _, c in tick.iterrows():
                options.append((SequenceMatcher(
                    None, mkey, c["key"]).ratio(), c, "ticker"))
            options.append((scores[best_i], crsp.loc[best_i], "name"))
            options.sort(key=lambda t: t[0], reverse=True)
            s, c, how = options[0]
            if contained(mtoks, c["toks"]) or s >= 0.92:
                cand, rule = pd.DataFrame([c]), f"{how}:{s:.2f}"
            else:
                rows.append({**m, "decision": "reject",
                             "rule": f"no-rule:{how}:{s:.2f}",
                             "best_comnam": c["comnam"],
                             "best_ticker": c["ticker"]})
                continue
        c = cand.iloc[0]
        rows.append({**m, "decision": "accept", "rule": rule,
                     "permno": int(c["permno"]),
                     "crsp_ticker": c["ticker"], "comnam": c["comnam"]})

    out = pd.DataFrame(rows)
    acc = out[out["decision"] == "accept"].copy()

    # permno -> gvkey
    lk = lk.drop_duplicates(["gvkey", "permno"])
    lk["permno"] = lk["permno"].astype(int)
    acc = acc.merge(lk[["gvkey", "permno"]], on="permno", how="left")
    no_gv = acc["gvkey"].isna()
    out.loc[out.index.isin(acc[no_gv].index), "decision"] = "reject-no-gvkey"
    acc = acc[~no_gv]

    # dedup: one redcode per gvkey (most obs), one gvkey per redcode
    acc = (acc.sort_values(["gvkey", "n"], ascending=[True, False])
              .drop_duplicates("gvkey"))
    acc = (acc.sort_values(["redcode", "n"], ascending=[True, False])
              .drop_duplicates("redcode"))

    audit = out[out["decision"] != "accept"]
    acc.to_csv(CACHE / "hy_universe_link.csv", index=False)
    audit.to_csv(CACHE / "hy_universe_audit.csv", index=False)
    print(f"accepted: {len(acc)} gvkeys / {acc['redcode'].nunique()} redcodes")
    print(f"rejected: {len(audit)} (audit file)")
    print("\nspread distribution of accepted (bp):")
    print(acc["med_bp"].describe().round(1).to_string())
    print("\nACCEPTED (eyeball):")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(acc[["mticker" if "mticker" in acc.columns else "ticker",
                   "shortname", "comnam", "crsp_ticker", "med_bp",
                   "rule"]].sort_values("med_bp", ascending=False)
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
