import json
from pathlib import Path
import httpx
import asyncio

async def run():
    gemini_dir = Path.home() / ".gemini"
    creds_path = gemini_dir / "oauth_creds.json"
    
    with open(creds_path) as f:
        token = json.load(f)["access_token"]
        
    headers = {
        "User-Agent": "antigravity/1.15.8 darwin/arm64",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }),
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    for proj in ["-", "", "neo"]:
        payload = {
            "model": "gemini-3.0-pro",
            "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "requestType": "agent",
            "userAgent": "antigravity"
        }
        if proj:
            payload["project"] = proj
            
        print(f"Testing sandbox with project='{proj}'...")
        async with httpx.AsyncClient() as c:
            r = await c.post("https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse",
                headers=headers, json=payload, timeout=10.0)
            if r.status_code == 200:
                print(f"SUCCESS with project={proj}")
                return
            else:
                print(f"Failed {r.status_code}: {r.text[:200]}")

if __name__ == "__main__":
    asyncio.run(run())
