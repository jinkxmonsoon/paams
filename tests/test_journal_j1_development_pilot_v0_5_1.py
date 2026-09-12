"""Mock-only transport repair tests for Journal J1 v0.5.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from btom_v2 import journal_j1_development_prompting_v0_4_2 as prompting
from btom_v2 import run_journal_j1_development_pilot_v0_5_0 as v050
from btom_v2 import run_journal_j1_development_pilot_v0_5_1 as v051

MANIFEST_SHA256 = "9a2867423ee2d57be484ed47cf2bdf113a2867ff911c99e442c983f145b01b4f"


def prompts_and_order():
    prompts = [row for row in prompting.build_prospective()[2] if row["candidate_id"] == 1]
    return prompts, v050.request_order(prompts)


def install_preflight_mocks(monkeypatch):
    prompts, order = prompts_and_order()
    monkeypatch.setattr(v051.frozen, "validate_prompt_freeze", lambda _path: ({"verified": True}, prompts, {"pilot_ready": True}))
    return order


def test_manifest_order_seed_map_and_body_are_unchanged():
    manifest = json.loads(v050.MANIFEST_PATH.read_text())
    assert hashlib.sha256(v050.MANIFEST_PATH.read_bytes()).hexdigest() == MANIFEST_SHA256
    _, order = prompts_and_order()
    assert order == manifest["frozen_request_order"] and len(order) == 144
    assert manifest["pair_seeds"] == {f"{row['variant_id']}|{row['M_R']}|{row['M_I']}": row["requested_seed"] for row in order}
    for prompt, seed in (("example", 1), ("second", 2147483646)):
        assert v051.request_body(prompt, seed) == v050.request_body(prompt, seed)


def test_transport_headers_have_only_established_addition():
    client = v051.Client("secret-value")
    assert client.headers() == {"Authorization": "Bearer secret-value", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": v051.USER_AGENT}
    assert v051.USER_AGENT == "Mozilla/5.0 (compatible; BToM-MAS/1.0.0; +https://github.com/jinkxmonsoon/paams)"


def test_cloudflare_classifier():
    assert v051.classify_transport(403, error="Cloudflare Error 1010") == "cloudflare_1010_client_signature"
    assert v051.classify_transport(403, {"error_name": "browser_signature_banned"}) == "cloudflare_1010_client_signature"
    assert v051.classify_transport(403, error="forbidden") == "http_error"
    assert v051.classify_transport(200) is None


class CloudflareClient:
    calls = 0
    def __init__(self, _secret): pass
    def call(self, _prompt, _seed):
        self.calls += 1
        return False, 403, None, "Cloudflare error_code 1010 browser_signature_banned", 0.01


class SuccessfulClient:
    calls = []
    def __init__(self, _secret): self.calls = []
    def call(self, prompt, seed):
        self.calls.append((prompt, seed))
        return True, 200, {"choices": [{"message": {"content": '{"action":"CONTINUE_TASK"}'}, "finish_reason": "stop"}], "x_groq": {"seed": seed}, "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}, None, 0.02


def test_cloudflare_preflight_stops_after_one(monkeypatch, tmp_path):
    install_preflight_mocks(monkeypatch)
    client = CloudflareClient("")
    result = v051.execute(tmp_path / "out", tmp_path, sleep=lambda _: None, client_factory=lambda _secret: client, environ={"GROQ_API_KEY": "secret-value"})
    assert result == 1 and client.calls == 1
    status = json.loads((tmp_path / "out/workflow_status.json").read_text())
    assert status["attempted_calls"] == 1 and status["complete_calls"] == status["behavioral_observations"] == 0
    assert status["api_failures"] == 1 and status["parse_failures"] == 0
    assert status["transport_classification"] == "cloudflare_1010_client_signature" and status["transport_failure"] is True and status["technical_pass"] is False


def test_successful_preflight_retained_and_not_repeated(monkeypatch, tmp_path):
    order = install_preflight_mocks(monkeypatch)
    client = SuccessfulClient("")
    result = v051.execute(tmp_path / "out", tmp_path, sleep=lambda _: None, client_factory=lambda _secret: client, environ={"GROQ_API_KEY": "secret-value"})
    assert result == 0 and len(client.calls) == 144
    assert client.calls[0][1] == order[0]["requested_seed"]
    assert client.calls.count(client.calls[0]) == 1
    records = [json.loads(line) for line in (tmp_path / "out/journal_j1_development_pilot_call_records_v0_5_1.jsonl").read_text().splitlines()]
    assert len(records) == 144 and records[0]["ordinal"] == 1 and records[0]["complete"]
    status = json.loads((tmp_path / "out/workflow_status.json").read_text())
    assert status["attempted_calls"] == status["complete_calls"] == status["behavioral_observations"] == 144
    assert status["behavioral_retries"] == status["fallback_actions"] == 0 and status["technical_pass"]


def test_provenance_and_secret_safety(monkeypatch, tmp_path):
    install_preflight_mocks(monkeypatch)
    client = CloudflareClient("")
    v051.execute(tmp_path / "out", tmp_path, sleep=lambda _: None, client_factory=lambda _secret: client, environ={"GROQ_API_KEY": "do-not-persist"})
    status = json.loads((tmp_path / "out/workflow_status.json").read_text())
    assert all(status[key] == value for key, value in v051.PREDECESSOR.items())
    assert status["scientific_protocol_changed"] is False and status["predecessor_behavioral_observations"] == 0
    assert "do-not-persist" not in "".join(path.read_text() for path in (tmp_path / "out").iterdir())
    source = Path(v051.__file__).read_text().lower()
    assert "120b" not in source and "confirmatory_prompts_generated" in source
