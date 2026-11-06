"""Workstream C: small-volume, re-runnable collection of the prospective SJF cohort.

Logic copied from the source-home census clients (read-only originals:
``C:\\uol_legacy_home\\src\\uol\\acquisition\\sjf_bicentenario_ids.py`` and
``sjf_bicentenario_content_full.py``): a temporary, normal Chrome profile opens the official public
entry page and performs same-origin GETs of the public JSON API (``/tesis/count``, ``/tesis/ids``
page 0 = newest ids, ``/tesis/<id>``). Challenges fail closed (no CAPTCHA or bot-check handling).
One request at a time with >= ``MIN_GAP`` seconds between requests. Cookies are not persisted.

New ids are those absent from the accepted 311,965-id census (14 Sep 2026). Raw response bodies
(public metadata/text only) are appended to ``prospective/raw/responses.jsonl`` with SHA-256;
already-fetched ids are skipped on re-runs, so a later cutoff (look 2: 2026-10-31) only fetches
new ids. ``build`` turns stored responses into v11-format source rows for a publication-date
window [start, cutoff].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from uol.facts_to_law.spec import sha256_file

from .c_sources import RUN, source_row

ENTRY_URL = "https://bicentenario.scjn.gob.mx/repositorio-scjn/sjf?fuente=Tesis&tab=3"
HOST = "bicentenario.scjn.gob.mx"
API = "/repositorio-scjn/api/v1/tesis"
CENSUS_IDS = Path(r"C:\uol_legacy_home\data\derived\corpus_execution_v2\sjf_bicentenario_ids_v2\ids.jsonl")
CENSUS_SHA256 = "61b8bd4df06f7fc01eb02af66e865a11db1f7eaa86360f2fca054a363139b725"
PAGE_SIZE = 1000
MIN_GAP = 1.2
MAX_DETAILS = 600           # hard cap per run (small volume)
CHALLENGE = ("captcha", "access denied", "request rejected", "verify you are human", "incapsula")
DIR = RUN / "prospective"
RAW = DIR / "raw"
FETCH = r"""
const done=arguments[arguments.length-1], url=arguments[0];
fetch(url,{method:'GET',cache:'no-store',credentials:'same-origin',headers:{'Accept':'application/json,text/plain,*/*'}})
 .then(async r=>{const b=await r.text();done({url,final_url:r.url,status:r.status,body:b});})
 .catch(e=>done({url,final_url:url,status:0,body:'',error:String(e)}));
"""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def challenge(text: str) -> str | None:
    t = str(text).casefold()
    if "incapsula" in t and "_incapsula_resource" in t and not any(x in t for x in ("blocked", "challenge", "denied", "rejected", "verify you are human")):
        t = t.replace("incapsula", "")
    return next((m for m in CHALLENGE if m in t), None)


class Session:
    def __init__(self) -> None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        self.profile = tempfile.TemporaryDirectory(prefix="uol-round2-c-")
        o = Options()
        o.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        for arg in (f"--user-data-dir={self.profile.name}", "--window-position=-32000,-32000", "--window-size=1280,900",
                    "--disable-gpu", "--no-first-run", "--disable-notifications"):
            o.add_argument(arg)
        self.driver = webdriver.Chrome(options=o)
        self.driver.set_page_load_timeout(45)
        self.driver.set_script_timeout(60)
        self.requests = 0
        self._last = 0.0
        self.driver.get(ENTRY_URL)
        time.sleep(2.0)
        reason = challenge(self.driver.page_source)
        if reason:
            self.close()
            raise RuntimeError(f"blocked: challenge marker {reason!r} on the entry page; stopping (no bypass)")

    def get(self, path: str) -> dict:
        wait = MIN_GAP - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        env = self.driver.execute_async_script(FETCH, path)
        self._last = time.monotonic()
        self.requests += 1
        final = urlsplit(str(env.get("final_url")))
        if (final.hostname or "").rstrip(".") != HOST:
            raise RuntimeError(f"off-origin redirect: {env.get('final_url')}")
        reason = challenge(env.get("body", ""))
        if reason:
            raise RuntimeError(f"blocked: challenge marker {reason!r} at {path}; stopping (no bypass)")
        return env

    def close(self) -> None:
        try:
            self.driver.quit()
        finally:
            self.profile.cleanup()


def census_ids() -> set[str]:
    if sha256_file(CENSUS_IDS) != CENSUS_SHA256:
        raise ValueError("census id file hash differs from the accepted census")
    with CENSUS_IDS.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def stored() -> dict[str, dict]:
    path = RAW / "responses.jsonl"
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if hashlib.sha256(r["body"].encode("utf-8")).hexdigest() != r["body_sha256"]:
                    raise ValueError(f"stored body hash mismatch for {r['id']}")
                out[r["id"]] = r
    return out


def collect(max_pages: int = 3) -> dict:
    """Fetch the newest-id pages, then every id absent from the census and from the raw store."""
    RAW.mkdir(parents=True, exist_ok=True)
    census, have = census_ids(), stored()
    t0, log = time.perf_counter(), {"started_utc": utc()}
    s = Session()
    try:
        env = s.get(f"{API}/count")
        count = json.loads(env["body"])
        log["count"] = count
        new_ids: list[str] = []
        with (RAW / "id_pages.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            for page in range(max_pages):
                env = s.get(f"{API}/ids?page={page}&size={PAGE_SIZE}")
                if env["status"] != 200:
                    raise RuntimeError(f"ids page {page}: HTTP {env['status']}")
                ids = [str(x) for x in json.loads(env["body"])]
                f.write(json.dumps({"utc": utc(), "page": page, "status": env["status"], "count": count,
                                    "body_sha256": hashlib.sha256(env["body"].encode()).hexdigest(), "ids": ids}) + "\n")
                fresh = [i for i in ids if i not in census]
                new_ids += fresh
                if len(fresh) < len(ids):      # reached ids already in the census
                    break
        todo = [i for i in dict.fromkeys(new_ids) if i not in have][:MAX_DETAILS]
        log["new_ids_vs_census"] = len(set(new_ids))
        log["fetched_now"] = 0
        with (RAW / "responses.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            for i in todo:
                env = s.get(f"{API}/{i}")
                body = env.get("body", "")
                f.write(json.dumps({"id": i, "utc": utc(), "url": env.get("final_url"), "status": env["status"],
                                    "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "body": body},
                                   ensure_ascii=False) + "\n")
                f.flush()
                log["fetched_now"] += 1
        log["requests"] = s.requests
    finally:
        s.close()
    log["wall_seconds"] = round(time.perf_counter() - t0, 1)
    log["finished_utc"] = utc()
    with (DIR / "collect_log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(log) + "\n")
    return log


def build(start: str, cutoff: str) -> dict:
    """v11-format source rows for stored responses with publication date in [start, cutoff]."""
    out = DIR / f"sources_{start}_{cutoff}.jsonl"
    raw_path = RAW / "responses.jsonl"
    file_sha = sha256_file(raw_path)
    reasons, rows, dates = Counter(), [], Counter()
    from uol.facts_to_law.sources import publication
    for i, r in sorted(stored().items(), key=lambda kv: int(kv[0])):
        if r["status"] != 200:
            reasons[f"http_{r['status']}"] += 1
            continue
        pub = json.loads(r["body"])
        if str(pub.get("idTesis")) != i:
            raise ValueError(f"response id mismatch for {i}")
        date, _ = publication(pub.get("notaPublica"))
        if date is None:
            reasons["no_publication_date"] += 1
            continue
        if not (start <= date <= cutoff):
            reasons["outside_window"] += 1
            continue
        dates[date] += 1
        row, why = source_row(pub, url=r["url"], line_sha256=r["body_sha256"], file_sha256=file_sha,
                              split="prospective", v1_component=None)
        reasons[why] += 1
        if row:
            rows.append(row)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"window": [start, cutoff], "created_utc": utc(), "raw_responses": {"path": str(raw_path), "sha256": file_sha},
                "reasons": dict(reasons), "rows": len(rows), "publication_dates_in_window": dict(sorted(dates.items())),
                "output": {"path": str(out), "sha256": sha256_file(out)}, "code": {"c_collect.py": sha256_file(Path(__file__))}}
    (DIR / f"sources_{start}_{cutoff}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["collect", "build"])
    p.add_argument("--start", default="2026-09-12")
    p.add_argument("--cutoff", default=datetime.now().date().isoformat())
    p.add_argument("--max-pages", type=int, default=3)
    a = p.parse_args()
    print(json.dumps(collect(a.max_pages) if a.command == "collect" else build(a.start, a.cutoff), indent=1))


if __name__ == "__main__":
    main()
