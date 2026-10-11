"""Small JSON API on the standard library (no web framework needed).

    POST /investigate            {"customer_id": "C0003"}
    GET  /cases[?status=OPEN]
    GET  /cases/<id>
    POST /cases/<id>/<action>    {"note": "..."}      action: start_review|escalate|close_no_action|approve_sar|close_escalated
    GET  /audit                  (needs view_audit)
    GET  /audit/verify           (needs view_audit)

Authenticate with ``Authorization: Bearer <token>``. Tokens map to (user, role) in the dict you pass
to ``make_server``. Demo tokens only: use a real identity provider in production.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agent import CaseAgent
from .cases import CaseService, WorkflowError
from .rbac import PermissionDenied, Principal, require


def make_server(agent: CaseAgent, tokens: dict[str, tuple[str, str]], host: str = "127.0.0.1", port: int = 0):
    service: CaseService = agent.service
    audit = service.audit

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, status: int, body: dict | list):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _principal(self) -> Principal | None:
            auth = self.headers.get("Authorization", "")
            entry = tokens.get(auth.removeprefix("Bearer ").strip()) if auth.startswith("Bearer ") else None
            return Principal(*entry) if entry else None

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                raise WorkflowError("request body must be valid JSON")

        def _dispatch(self, method: str):
            p = self._principal()
            if p is None:
                return self._send(401, {"error": "missing or invalid token"})
            url = urlparse(self.path)
            parts = [s for s in url.path.split("/") if s]
            query = parse_qs(url.query)
            try:
                if method == "POST" and parts == ["investigate"]:
                    cid = self._body().get("customer_id")
                    if cid not in agent.data.customers:
                        return self._send(404, {"error": f"unknown customer '{cid}'"})
                    result = agent.investigate(p, cid)
                    result["trace"] = [t["tool"] for t in result["trace"]]
                    return self._send(200, result)
                if method == "GET" and parts == ["cases"]:
                    return self._send(200, service.list_cases(p, (query.get("status") or [None])[0]))
                if method == "GET" and len(parts) == 2 and parts[0] == "cases":
                    return self._send(200, service.view_case(p, parts[1]))
                if method == "POST" and len(parts) == 3 and parts[0] == "cases":
                    return self._send(200, service.act(p, parts[1], parts[2], self._body().get("note", "")))
                if method == "GET" and parts == ["audit"]:
                    require(p, "view_audit")
                    return self._send(200, audit.entries(limit=int((query.get("limit") or [100])[0])))
                if method == "GET" and parts == ["audit", "verify"]:
                    require(p, "view_audit")
                    ok, bad = audit.verify()
                    return self._send(200, {"ok": ok, "first_bad_seq": bad, "entries": len(audit)})
                return self._send(404, {"error": "not found"})
            except PermissionDenied as exc:
                audit.append(p.user, p.role, f"http.{method} {url.path}", "-", "denied", {"permission": exc.permission})
                return self._send(403, {"error": str(exc)})
            except WorkflowError as exc:
                return self._send(409, {"error": str(exc)})
            except KeyError as exc:
                return self._send(404, {"error": f"not found: {exc}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:   # pragma: no cover
    from .audit import AuditLog
    from .data import generate_dataset

    agent = CaseAgent(generate_dataset(), CaseService(AuditLog()))
    tokens = {"analyst-token": ("asha", "analyst"), "officer-token": ("ravi", "compliance_officer"),
              "auditor-token": ("meera", "auditor"), "admin-token": ("dev", "admin")}
    server = make_server(agent, tokens, port=8000)
    print("BankComply API on http://127.0.0.1:8000 (demo tokens: analyst-token, officer-token, auditor-token, admin-token)")
    server.serve_forever()


if __name__ == "__main__":   # pragma: no cover
    main()
