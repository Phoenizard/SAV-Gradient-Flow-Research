"""Slack webhook notification utility for SAV-Gradient-Flow-Research."""

import json
import os
import urllib.request
import urllib.error

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def send_slack(message: str) -> bool:
    """Send a message to the configured Slack webhook.

    Set the SLACK_WEBHOOK_URL environment variable before use.
    Returns True on success, False on failure (never raises).
    """
    if not SLACK_WEBHOOK_URL:
        print("[slack] SLACK_WEBHOOK_URL not set, skipping notification.")
        return False
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[slack] Failed to send notification: {e}")
        return False
