# Chatting with Qelvari from Anywhere (Antigravity Remote Agent & Remote Control)

You can access, prompt, test, and deploy **Qelvari** from any laptop, tablet, or phone browser at **https://antigravity.google.com** using either **Remote Agents** (cloud VM, works even when your PC is off) or **Remote Control** (drives your PC workspace live).

---

## Option 1: Antigravity Remote Agents (Cloud VM — Works With Your PC Off)

Remote Agents run in an isolated Google-managed Linux VM connected directly to your GitHub repository (`0xxay/Qelvari`). Whenever a session starts, Antigravity clones the repository and automatically reads [`AGENTS.md`](../AGENTS.md) so the agent already knows Qelvari's architecture, setup commands, and test suite.

### Setup from Any Web or Mobile Browser (Recommended)
1. Open **https://antigravity.google.com** on your phone or browser and sign in with your Google Account (requires Google AI Pro or Google AI Ultra).
2. In the prompt box, click the **Project Selector** dropdown -> choose **Remote** -> **GitHub Repositories** (or click **+** next to **Projects** -> **New Remote Project** -> **Add Resource** -> **GitHub**).
3. Authorize GitHub (first time only) and select **`0xxay/Qelvari`**.
4. Send any prompt (for example: *"Run the Qelvari test suite and benchmark FLAME against backdoor poisoning"*). The Remote Agent will work in the cloud even if you lock your phone or close the tab, and can push branches or open Pull Requests directly.

### Setup from Terminal (`agy` CLI)
```bash
# Start or resume a cloud Remote Agent session from anywhere:
agy --remote-agent --new-project
# Resume your latest remote conversation later:
agy --remote-agent -c
```

---

## Option 2: Antigravity Remote Control (Drive Your Local PC From Anywhere)

Use **Remote Control** when your PC is powered on and online and you want to continue chatting with your exact local `d:\ppflids` workspace, local terminals, and uncommitted files from your phone or another browser.

### A. From the Antigravity Desktop App (2 Clicks)
1. Press **`Ctrl + ,`** in Antigravity to open **Settings** -> **Application**.
2. Toggle **Enable Remote Control** to **On** (and optionally set Device Name to `qelvari-pc`).
3. Open **https://antigravity.google.com** on your phone or laptop (tip: tap *Add to Home Screen* on mobile for push notifications when tasks finish) and pick `qelvari-pc` in the **Instance Switcher**.

### B. Always-On Background CLI Daemon (Windows Service / Scheduled Task)
```powershell
# Register & start the persistent background daemon on your PC:
agy remote-control start --name "qelvari-workstation"

# Check connection status & assigned instance name:
agy remote-control status
```
Once running, select `qelvari-workstation` in the instance switcher at **https://antigravity.google.com** anytime your PC is online.
