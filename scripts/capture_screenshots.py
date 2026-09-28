"""
scripts/capture_screenshots.py
==============================
Automated screenshot generator for VoiceFinAI documentation.
Uses Playwright with headless Chrome to capture crisp, high-res UI previews.
"""

import os
import time
from playwright.sync_api import sync_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "assets"))
os.makedirs(ASSETS_DIR, exist_ok=True)


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)

        # 1. Hero Interface
        print("Capturing 1. Hero interface...")
        page = browser.new_page(viewport={"width": 1280, "height": 840})
        page.goto("http://127.0.0.1:5000")
        time.sleep(1)
        page.screenshot(path=os.path.join(ASSETS_DIR, "hero_interface.png"))

        # 2. Fund recommendations
        print("Capturing 2. Fund recommendations...")
        page.fill("#queryInput", "Show me moderate risk mutual funds")
        page.click(".send-button")
        page.wait_for_selector(".fund-card", timeout=15000)
        time.sleep(1.5)
        page.screenshot(path=os.path.join(ASSETS_DIR, "fund_recommendations.png"))

        # 3. What-If Simulator & Performance Chart
        print("Capturing 3. Performance & SIP projection...")
        page.fill("#queryInput", "chart dikhao")
        page.click(".send-button")
        page.wait_for_selector("#chartContainer .whatif-panel", timeout=15000)
        time.sleep(1.5)
        page.evaluate("() => { const el = document.getElementById('conversationBottom'); if(el) el.scrollTop = 220; }")
        time.sleep(0.8)
        page.screenshot(path=os.path.join(ASSETS_DIR, "sip_calculator_chart.png"))

        # 4. Goal-Based Investing in a fresh page
        print("Capturing 4. Goal planner...")
        page2 = browser.new_page(viewport={"width": 1280, "height": 920})
        page2.goto("http://127.0.0.1:5000")
        time.sleep(0.5)
        page2.fill("#queryInput", "Plan a SIP for 25 lakh in 5 years")
        page2.click(".send-button")
        page2.wait_for_selector("#goalContainer .goal-header-card", timeout=15000)
        time.sleep(1.5)
        page2.evaluate("() => { const el = document.getElementById('conversationBottom'); if(el) el.scrollTop = 160; }")
        time.sleep(0.8)
        page2.screenshot(path=os.path.join(ASSETS_DIR, "goal_planner.png"))
        page2.close()

        page.close()
        browser.close()
        print(f"All screenshots saved to: {ASSETS_DIR}")


if __name__ == "__main__":
    capture_all()
