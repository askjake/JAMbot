#!/usr/bin/env bash
set -euo pipefail
REMOTE="montjac@10.79.85.47"
REMOTE_DIR="~/sling"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "=== [1/6] Writing files to $TMP ==="

# ─── app.py ────────────────────────────────────────────────────────────────
cat > "$TMP/app.py" << 'PYEOF'
import os
import time
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    layout="wide",
    page_title="Sling Live",
    page_icon="📡",
)

STREAMS   = [f"live{i}" for i in range(9)]
HLS_BASE  = "http://10.79.85.47:8088/hls"
HLS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hls_segments")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Sling Live")
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    st.markdown("**Stream Status**")
    for name in STREAMS:
        m3u8  = os.path.join(HLS_DIR, name, "index.m3u8")
        alive = os.path.exists(m3u8) and (time.time() - os.path.getmtime(m3u8)) < 10
        dot   = "🟢" if alive else "⚫"
        st.markdown(f"{dot} `{name}`")
    st.markdown("---")
    st.caption("Streamlit :8501  |  HLS :8088")

st.markdown("## 📡 Sling Live — 9 Stream Monitor")

# ── HLS player URLs ───────────────────────────────────────────────────────────
stream_urls = [f"{HLS_BASE}/{n}/index.m3u8" for n in STREAMS]

# ── Per-player JS ─────────────────────────────────────────────────────────────
players_js = ""
for i, (name, url) in enumerate(zip(STREAMS, stream_urls)):
    players_js += f"""
(function(){{
    var video  = document.getElementById('v{i}');
    var ph     = document.getElementById('ph{i}');
    var src    = '{url}';
    var retry  = null;
    function load(){{
        if(Hls.isSupported()){{
            if(video._hls){{ video._hls.destroy(); }}
            var hls = new Hls({{
                liveSyncDurationCount:3,
                liveMaxLatencyDurationCount:6,
                maxBufferLength:8,
                enableWorker:true
            }});
            video._hls = hls;
            hls.loadSource(src);
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED, function(){{
                video.play().catch(function(){{}});
                ph.style.display='none';
            }});
            hls.on(Hls.Events.ERROR, function(e,d){{
                if(d.fatal){{
                    ph.style.display='flex';
                    hls.destroy();
                    if(!retry) retry=setTimeout(function(){{
                        retry=null; ph.style.display='flex'; load();
                    }},3000);
                }}
            }});
        }} else if(video.canPlayType('application/vnd.apple.mpegurl')){{
            video.src=src; video.play().catch(function(){{}});
            ph.style.display='none';
        }} else {{
            ph.querySelector('span').textContent='{name} — HLS not supported';
        }}
    }}
    load();
}})();
"""

# ── Cell HTML ─────────────────────────────────────────────────────────────────
cells = ""
for i, name in enumerate(STREAMS):
    cells += f"""
<div class="cell">
  <div class="wrap">
    <video id="v{i}" muted playsinline preload="none"></video>
    <div class="ph" id="ph{i}"><span>📡 {name} — waiting…</span></div>
  </div>
  <div class="lbl">{name}</div>
</div>
"""

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0e1117;color:#fff;font-family:sans-serif;overflow:hidden}}
.grid{{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  grid-template-rows:repeat(3,1fr);
  gap:5px;width:100%;height:100vh;padding:4px;
}}
.cell{{display:flex;flex-direction:column;background:#1a1d27;border-radius:6px;overflow:hidden;border:1px solid #2d3147}}
.wrap{{position:relative;flex:1;background:#000}}
video{{width:100%;height:100%;object-fit:cover;display:block}}
.ph{{display:flex;position:absolute;inset:0;align-items:center;justify-content:center;background:#0a0c14;color:#555;font-size:12px}}
.lbl{{text-align:center;padding:3px 0;font-size:11px;color:#888;background:#12141f;letter-spacing:.5px;text-transform:uppercase}}
</style></head><body>
<div class="grid">{cells}</div>
<script>{players_js}</script>
</body></html>"""

components.html(html, height=800, scrolling=False)

with st.expander("Stream URLs"):
    for name, url in zip(STREAMS, stream_urls):
        st.code(f"rtmp://10.79.86.118:1935/{name}  →  {url}")
PYEOF

# ─── hls_server.py ─────────────────────────────────────────────────────────
cat > "$TMP/hls_server.py" << 'PYEOF'
#!/usr/bin/env python3
"""CORS-enabled HLS HTTP server. Usage: python3 hls_server.py <dir> [port]"""
import sys, os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()
    def do_OPTIONS(self):
        self.send_response(200); self.end_headers()
    def log_message(self, fmt, *a): pass

if __name__ == "__main__":
    d    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "hls_segments")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8088
    os.makedirs(d, exist_ok=True)
    os.chdir(d)
    srv = HTTPServer(("0.0.0.0", port), CORSHandler)
    print(f"[hls_server] Serving {d} on :{port}", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: print("[hls_server] Stopped.")
PYEOF

# ─── start.sh ──────────────────────────────────────────────────────────────
cat > "$TMP/start.sh" << 'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HLS="$BASE/hls_segments"
LOG="$BASE/logs"
PID="$BASE/pids"
RTMP="rtmp://10.79.86.118:1935"

mkdir -p "$LOG" "$PID"

echo "=== Sling Live — Starting ==="

# Stop old processes
echo "[*] Cleaning up existing processes..."
pkill -f "hls_server.py"        2>/dev/null || true
pkill -f "streamlit run"        2>/dev/null || true
pkill -f "10.79.86.118:1935"    2>/dev/null || true
sleep 1
fuser -k 8088/tcp 2>/dev/null || true
fuser -k 8501/tcp 2>/dev/null || true
sleep 1

# Create HLS output dirs
for i in $(seq 0 8); do mkdir -p "$HLS/live${i}"; done

# HLS HTTP server
echo "[*] Starting HLS server on :8088..."
nohup python3 "$BASE/hls_server.py" "$HLS" 8088 \
    > "$LOG/hls_server.log" 2>&1 &
echo $! > "$PID/hls_server.pid"
sleep 1

# FFmpeg — one per stream
for i in $(seq 0 8); do
    NAME="live${i}"
    echo "[*] FFmpeg → $NAME"
    nohup ffmpeg -re \
        -i "${RTMP}/${NAME}" \
        -c:v libx264 -preset ultrafast -tune zerolatency \
        -b:v 1500k -maxrate 1500k -bufsize 3000k \
        -c:a aac -b:a 128k -ar 44100 \
        -f hls \
        -hls_time 2 \
        -hls_list_size 3 \
        -hls_flags delete_segments+append_list \
        -hls_segment_filename "$HLS/${NAME}/seg%03d.ts" \
        "$HLS/${NAME}/index.m3u8" \
        > "$LOG/ffmpeg${i}.log" 2>&1 &
    echo $! > "$PID/ffmpeg${i}.pid"
done

echo "[*] Waiting for FFmpeg to initialize..."
sleep 4

# Streamlit
echo "[*] Starting Streamlit on :8501..."
nohup streamlit run "$BASE/app.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$LOG/streamlit.log" 2>&1 &
echo $! > "$PID/streamlit.pid"

sleep 2
echo ""
echo "✅ Sling Live is up!"
IP=$(hostname -I | awk '{print $1}')
echo "   Dashboard : http://${IP}:8501"
echo "   HLS server: http://${IP}:8088"
echo ""
echo "Verify HLS segments: ls $HLS/live0/"
echo "Stop with          : $BASE/stop.sh"
SHEOF

# ─── stop.sh ───────────────────────────────────────────────────────────────
cat > "$TMP/stop.sh" << 'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID="$BASE/pids"

stop_pid() {
    [[ -f "$1" ]] || return 0
    local p; p=$(cat "$1")
    kill -0 "$p" 2>/dev/null && { echo "[*] Stopping $2 (PID $p)"; kill "$p" 2>/dev/null || true; }
    rm -f "$1"
}

echo "=== Sling Live — Stopping ==="
stop_pid "$PID/streamlit.pid"   "Streamlit"
stop_pid "$PID/hls_server.pid"  "HLS server"
for i in $(seq 0 8); do stop_pid "$PID/ffmpeg${i}.pid" "FFmpeg live${i}"; done

pkill -f "hls_server.py"      2>/dev/null || true
pkill -f "streamlit run"      2>/dev/null || true
pkill -f "10.79.86.118:1935"  2>/dev/null || true

echo "✅ Stopped."
SHEOF

# ─── requirements.txt ──────────────────────────────────────────────────────
cat > "$TMP/requirements.txt" << 'EOF'
streamlit>=1.32.0
EOF

