# 🚀 Deployment Guide

This guide covers running the Pneumonia Detection system locally, sharing it over a LAN, and exposing it securely to external users via Cloudflare Tunnel.

---

## Local Deployment

### 1. Start the Streamlit Dashboard

```bash
# From the project root, with the virtual environment active
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 2. Run in Headless Mode (WSL2 / Servers)

WSL2 cannot launch a GUI browser automatically. Run in headless mode instead:

```bash
streamlit run app.py --server.headless true
```

Or make it permanent via `.streamlit/config.toml`:

```toml
[server]
headless = true
```

---

## Ollama Setup (AI Report Generation)

The AI report feature requires a local Ollama instance with `qwen2.5` loaded.

### Windows (PowerShell)

```powershell
# Allow WSL2 connections by binding to all interfaces
$env:OLLAMA_HOST = "0.0.0.0"
ollama serve

# In a separate terminal — pull the model (one-time)
ollama pull qwen2.5
```

> **Why `OLLAMA_HOST=0.0.0.0`?**  
> By default Ollama only listens on `127.0.0.1`. WSL2 runs in a separate VM and reaches Windows through a virtual gateway IP (e.g., `172.x.x.x`). Setting `0.0.0.0` makes Ollama accept connections from all interfaces, including WSL2.

### Linux / WSL2

```bash
# Run Ollama on CPU to preserve GPU VRAM for TensorFlow
CUDA_VISIBLE_DEVICES="" ollama serve &
ollama pull qwen2.5
```

---

## LAN Access

To let other devices on your local network access the dashboard:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Other devices can connect at `http://<your-machine-ip>:8501`.

Find your machine's IP:
```bash
# Linux / WSL2
ip route get 1 | awk '{print $7; exit}'

# Windows PowerShell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike '*Loopback*' }).IPAddress
```

---

## Cloudflare Tunnel (External Access)

Share the dashboard publicly over HTTPS with no firewall changes, no port forwarding, and no static IP.

### Quick Tunnel (No Account — Temporary URL)

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Start the Streamlit server
streamlit run app.py --server.headless true

# Open a public HTTPS tunnel in a second terminal
cloudflared tunnel --url http://localhost:8501
```

A temporary URL is generated (e.g., `https://random-name.trycloudflare.com`). Share it with anyone — it's end-to-end encrypted and requires no account. The URL expires when you close the tunnel.

---

### Named Tunnel (Persistent — Custom Domain)

For a permanent deployment under your own domain with Cloudflare Access policies:

#### Step 1 — Authenticate

```bash
cloudflared tunnel login
```

This opens a browser and links `cloudflared` to your Cloudflare account.

#### Step 2 — Create a Named Tunnel

```bash
cloudflared tunnel create pneumonia-ai
```

This creates a tunnel with a stable UUID and stores credentials in `~/.cloudflared/`.

#### Step 3 — Route a Domain

```bash
# Replace with your actual domain managed by Cloudflare
cloudflared tunnel route dns pneumonia-ai pneumonia-ai.yourdomain.com
```

#### Step 4 — Create a Config File

```yaml
# ~/.cloudflared/config.yml
tunnel: pneumonia-ai
credentials-file: /home/<user>/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: pneumonia-ai.yourdomain.com
    service: http://localhost:8501
  - service: http_status:404
```

#### Step 5 — Run the Tunnel

```bash
cloudflared tunnel run pneumonia-ai
```

---

### Restricting Access with Cloudflare Access

To require login before accessing the dashboard (recommended for medical data):

1. Go to **Cloudflare Zero Trust → Access → Applications**
2. Add an application pointing to your tunnel hostname
3. Set an access policy (e.g., allow only specific email addresses or a GitHub organization)

Users will be prompted to authenticate via email OTP or SSO before reaching the Streamlit app.

---

## Environment Variables Reference

| Variable | Scope | Description |
|---|---|---|
| `OLLAMA_HOST` | Windows PowerShell | Bind address for Ollama (set to `0.0.0.0` for WSL2 access) |
| `CUDA_VISIBLE_DEVICES` | Linux / WSL2 | Set to `""` to force Ollama onto CPU, freeing VRAM for TensorFlow |
| `TF_ENABLE_ONEDNN_OPTS` | Linux / WSL2 | Set to `0` to suppress oneDNN info messages |
