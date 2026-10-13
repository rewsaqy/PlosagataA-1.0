"""Konektor Context7 untuk Sagi — server memanggil docs sebagai fungsi.

Model kecil tidak bisa protokol MCP sendiri; script ini yang bicara MCP
(Streamable HTTP) memakai kredensial opencode.json milikmu, lalu hasilnya
bisa disuntik ke konteks Sagi / dipakai web.

Pakai:
  python scripts/ctx7.py "pytorch" "how to use DataLoader batching"
"""
import json
import sys
import urllib.request
from pathlib import Path


def _cfg():
    d = json.loads(Path.home().joinpath(".config/opencode/opencode.json").read_text())
    c = d["mcp"]["context7"]
    return c["url"], c["headers"]["Authorization"]


class Ctx7:
    def __init__(self):
        self.url, auth = _cfg()
        self.base_headers = {
            "Authorization": auth,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self.session = None
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "sagi", "version": "1.0"},
        })
        self._rpc("notifications/initialized", None, notify=True)

    def _rpc(self, method, params, notify=False):
        body = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            body["id"] = 1
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers=self.base_headers)
        if self.session:
            req.add_header("Mcp-Session-Id", self.session)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.session = r.headers.get("Mcp-Session-Id", self.session)
            raw = r.read().decode()
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = json.loads(line[5:])
                if "result" in payload:
                    return payload["result"]
                return payload
        try:
            payload = json.loads(raw)
            return payload.get("result", payload)
        except Exception:
            return {"raw": raw[:500]}

    def call(self, tool, args):
        return self._rpc("tools/call", {"name": tool, "arguments": args})

    def docs(self, library, query):
        import re
        libs = self.call("resolve-library-id", {"libraryName": library, "query": query})
        blob = ""
        if isinstance(libs, dict):
            for c in libs.get("content") or []:
                if c.get("type") == "text":
                    blob += c["text"] + "\n"
        m = re.search(r"Context7-compatible library ID:\s*(/\S+)", blob)
        if not m:
            m = re.search(r"(/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)", blob)
        if not m:
            return f"Tidak ketemu library {library}. Mentah: {blob[:300]}"
        libid = m.group(1).strip()
        out = self.call("query-docs", {"libraryId": libid, "query": query})
        texts = []
        for c in (out.get("content") or []):
            if c.get("type") == "text":
                texts.append(c["text"])
        return "\n\n".join(texts)[:4000] or str(out)[:1000]


if __name__ == "__main__":
    lib = sys.argv[1] if len(sys.argv) > 1 else "PyTorch"
    q = sys.argv[2] if len(sys.argv) > 2 else "DataLoader batching"
    print(Ctx7().docs(lib, q))
