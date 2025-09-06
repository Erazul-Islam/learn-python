#!/usr/bin/env python3
"""
Big AI Chatbot (No External Libraries)
--------------------------------------
A single-file, BBA-friendly chatbot that runs in the terminal.
- No pip installs required (built-in Python only)
- Intent detection with regex + keyword scoring
- Sentiment analysis (simple lexicon)
- Multi-turn slot filling (e.g., complaint ticket)
- FAQ for business topics (BBA-focused)
- Safe calculator (AST-based)
- CSV quick insights (read local CSV, compute stats)
- Memory persistence to JSON (remembers your name & preferences)
- Commands: /help, /reset, /export_history, /summary, /loadcsv path, /setname Name

Run:
    python big_ai_chatbot.py

"""
from __future__ import annotations
import re
import json
import os
import sys
import random
import datetime as dt
import statistics as stats
import csv
import ast
from typing import Any, Dict, List, Tuple, Optional

# ----------------------------- Utilities ----------------------------- #

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize(text: str) -> str:
    text = text.strip().lower()
    # keep basic punctuation for intent detection but normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------------- Safe Calculator --------------------------- #
# Evaluate arithmetic expressions safely using AST; supports + - * / ** % ( )
class SafeEvaluator(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Name
    )
    ALLOWED_NAMES = {
        # optional handy constants
        'pi': 3.141592653589793,
        'e': 2.718281828459045,
    }

    def visit(self, node):  # type: ignore[override]
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError("Disallowed expression")
        return super().visit(node)

    def eval(self, expr: str) -> float:
        tree = ast.parse(expr, mode='eval')
        return self._eval(tree.body)

    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
        if isinstance(node, ast.Name):
            if node.id in self.ALLOWED_NAMES:
                return self.ALLOWED_NAMES[node.id]
            raise ValueError("Unknown name: %s" % node.id)
        raise ValueError("Unsupported expression")


# ----------------------- Simple Sentiment ---------------------------- #
POS_WORDS = set(
    """
    good great excellent awesome amazing love like happy satisfied helpful fast
    fantastic superb brilliant wonderful recommend positive smooth convenient
    affordable reasonable polite friendly quick impressive neat clean reliable
    """.split()
)
NEG_WORDS = set(
    """
    bad poor terrible awful hate dislike unhappy unsatisfied slow rude broken
    worst late expensive dirty confusing frustrating unhelpful problem issue
    negative buggy crash delay
    """.split()
)


def sentiment_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    score = 0
    for t in tokens:
        if t in POS_WORDS:
            score += 1
        elif t in NEG_WORDS:
            score -= 1
    return score / max(1, len(tokens))


# -------------------------- Knowledge Base -------------------------- #
FAQ = {
    # Finance & Accounting
    "sunk cost": "Sunk cost is money already spent and unrecoverable; ignore it when making future decisions.",
    "opportunity cost": "Opportunity cost is the value of the next best alternative you give up when making a choice.",
    "npv": "NPV (Net Present Value) = sum of discounted cash flows minus initial investment; choose projects with positive NPV.",
    "irr": "IRR is the discount rate that makes NPV = 0; select projects with IRR above the required return.",
    "capm": "CAPM: Expected Return = Rf + Beta * (Rm - Rf).",
    "beta": "Beta measures a stock's sensitivity to market movements (beta > 1 more volatile than market).",
    "perpetuity": "Perpetuity PV = C / r (first payment one period from now).",
    # Marketing
    "4p": "Marketing Mix 4P: Product, Price, Place, Promotion.",
    "stp": "STP: Segmentation, Targeting, Positioning.",
    "swot": "SWOT: Strengths, Weaknesses, Opportunities, Threats.",
    # Management & Ops
    "kpi": "KPI: Key Performance Indicator—quantifiable measure of performance over time.",
    "lean": "Lean aims to eliminate waste (muda) and maximize customer value.",
    "six sigma": "Six Sigma reduces process variation; DMAIC: Define, Measure, Analyze, Improve, Control.",
}

SMALLTALK = [
    "Totally noted.",
    "Got it!",
    "Interesting—tell me more.",
    "Thanks for sharing!",
    "I'm here to help—what's next?",
]

GREETINGS = [
    "Hello! I'm your study buddy bot. What's your name?",
    "Hi there! Great to see you. How can I help today?",
    "Hey! Ready to learn something new?",
]

GOODBYES = [
    "Bye! Keep learning and stay awesome!",
    "See you later—good luck with your studies!",
    "Goodbye! Ping me anytime.",
]

# ----------------------------- Memory ------------------------------- #
class Memory:
    def __init__(self, path: str = "bot_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {
            "user_name": None,
            "preferences": {},
            "history": [],  # list of {time, role, text}
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_history(self, role: str, text: str):
        self.data["history"].append({"time": now_str(), "role": role, "text": text})
        # Keep last 200 turns
        if len(self.data["history"]) > 400:
            self.data["history"] = self.data["history"][-400:]
        self.save()


# ---------------------------- NLU/NLP ------------------------------- #
INTENT_PATTERNS = [
    ("greet", re.compile(r"\b(hi|hello|hey|assalam|salam)\b", re.I)),
    ("goodbye", re.compile(r"\b(bye|goodbye|see you|tata)\b", re.I)),
    ("set_name", re.compile(r"\bmy name is (.+)", re.I)),
    ("ask_time", re.compile(r"\b(time|clock|what time)\b", re.I)),
    ("ask_date", re.compile(r"\b(date|today)\b", re.I)),
    ("calculator", re.compile(r"^=|\b(calc|calculate|evaluate)\b", re.I)),
    ("faq", re.compile(r"\b(sunk cost|opportunity cost|npv|irr|capm|beta|perpetuity|4p|stp|swot|kpi|lean|six sigma)\b", re.I)),
    ("load_csv", re.compile(r"^/loadcsv\s+(.+)$", re.I)),
    ("setname_cmd", re.compile(r"^/setname\s+(.+)$", re.I)),
    ("help", re.compile(r"^/help$", re.I)),
    ("reset", re.compile(r"^/reset$", re.I)),
    ("export", re.compile(r"^/export_history$", re.I)),
    ("summary", re.compile(r"^/summary$", re.I)),
    ("complaint", re.compile(r"\b(complain|issue|problem|refund|return|support)\b", re.I)),
]


def detect_intent(text: str) -> Tuple[str, Optional[re.Match]]:
    for name, pattern in INTENT_PATTERNS:
        m = pattern.search(text)
        if m:
            return name, m
    # fallback heuristics
    if any(k in text for k in ("help", "commands")):
        return "help", None
    return "smalltalk", None


# -------------------------- Chatbot Core --------------------------- #
class Chatbot:
    def __init__(self):
        self.mem = Memory()
        self.safe_eval = SafeEvaluator()
        self.pending_task: Optional[Dict[str, Any]] = None  # for slot filling
        self.loaded_csv: Optional[str] = None
        self.csv_headers: List[str] = []
        self.csv_rows: List[Dict[str, Any]] = []

    # --------------- Reply helpers --------------- #
    def reply(self, text: str) -> str:
        self.mem.add_history("assistant", text)
        return text

    def say(self, text: str):
        print(text)
        self.mem.add_history("assistant", text)

    # --------------- Handlers --------------- #
    def handle_greet(self) -> str:
        name = self.mem.data.get("user_name")
        if name:
            return self.reply(f"Hello {name}! How can I help today?")
        return self.reply(random.choice(GREETINGS))

    def handle_goodbye(self) -> str:
        return self.reply(random.choice(GOODBYES))

    def handle_set_name(self, m: re.Match) -> str:
        name = m.group(1).strip().split()[0]
        self.mem.data["user_name"] = name.title()
        self.mem.save()
        return self.reply(f"Nice to meet you, {self.mem.data['user_name']}! I'll remember your name.")

    def handle_setname_cmd(self, m: re.Match) -> str:
        name = m.group(1).strip()
        self.mem.data["user_name"] = name.title()
        self.mem.save()
        return self.reply(f"Got it! I'll call you {self.mem.data['user_name']}.")

    def handle_time(self) -> str:
        return self.reply(f"Current time: {now_str()}")

    def handle_date(self) -> str:
        today = dt.date.today().strftime("%A, %B %d, %Y")
        return self.reply(f"Today is {today}.")

    def handle_help(self) -> str:
        txt = (
            "Commands:\n"
            "  /help                Show this help\n"
            "  /setname Name        Save your name\n"
            "  /loadcsv path.csv    Load a CSV and get insights\n"
            "  /summary             Summarize our recent chat\n"
            "  /export_history      Save conversation to chat_history.txt\n"
            "  /reset               Clear memory (name & history)\n\n"
            "Other features:\n"
            "- Type 'sunk cost', 'NPV', 'CAPM', etc. for quick BBA FAQs\n"
            "- Start a complaint (say 'I have a problem...') to open a support ticket\n"
            "- Calculator: start with '=' or type 'calculate 2*(10+5)'\n"
            "- Ask date/time, casual chat, and more!"
        )
        return self.reply(txt)

    def handle_export(self) -> str:
        path = "chat_history.txt"
        with open(path, "w", encoding="utf-8") as f:
            for item in self.mem.data.get("history", []):
                f.write(f"[{item['time']}] {item['role'].upper()}: {item['text']}\n")
        return self.reply(f"History exported to {path} (in the current folder).")

    def handle_summary(self) -> str:
        hist = self.mem.data.get("history", [])[-20:]
        # simple extractive summary: pick assistant/user highlights
        user_msgs = [h["text"] for h in hist if h["role"] == "user"]
        intents = []
        for u in user_msgs:
            intent, _ = detect_intent(normalize(u))
            intents.append(intent)
        intent_counts = {i: intents.count(i) for i in set(intents)}
        bullets = ", ".join(f"{k}×{v}" for k, v in intent_counts.items()) or "varied topics"
        return self.reply(f"Recent summary: we discussed {bullets}. I also saved your name if you set it.")

    def handle_reset(self) -> str:
        self.mem.data = {"user_name": None, "preferences": {}, "history": []}
        self.mem.save()
        return self.reply("Memory cleared. Fresh start!")

    def handle_calculator(self, text: str) -> str:
        # extract expression after '=' if present
        expr = text
        if text.strip().startswith("="):
            expr = text.strip()[1:]
        expr = re.sub(r"^(calc(ulate)?|evaluate)\s+", "", expr, flags=re.I)
        try:
            val = self.safe_eval.eval(expr)
            return self.reply(f"Result: {val}")
        except Exception as e:
            return self.reply(f"Sorry, I couldn't evaluate that. ({e})")

    def handle_faq(self, text: str) -> str:
        for k, v in FAQ.items():
            if k in text:
                return self.reply(f"{k.title()}: {v}")
        return self.reply("I didn't find that topic. Try /help for supported FAQs.")

    # ---- CSV quick insights ---- #
    def handle_load_csv(self, m: re.Match) -> str:
        path = m.group(1).strip().strip('"')
        if not os.path.exists(path):
            return self.reply("CSV not found. Please provide a valid path.")
        self.loaded_csv = path
        self.csv_headers = []
        self.csv_rows = []
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            self.csv_headers = reader.fieldnames or []
            for row in reader:
                self.csv_rows.append(row)
        if not self.csv_rows:
            return self.reply(f"Loaded {path}, but it has no rows.")
        msg = [f"Loaded {path} with {len(self.csv_rows)} rows and {len(self.csv_headers)} columns."]
        # Basic numeric stats for first numeric column
        numeric_cols = []
        for h in self.csv_headers:
            try:
                float(next((r[h] for r in self.csv_rows if r.get(h) not in (None, '',) ), 'nan'))
                numeric_cols.append(h)
            except Exception:
                continue
        if numeric_cols:
            col = numeric_cols[0]
            vals: List[float] = []
            for r in self.csv_rows:
                try:
                    vals.append(float(r[col]))
                except Exception:
                    pass
            if vals:
                msg.append(
                    f"Quick stats for '{col}': count={len(vals)}, mean={stats.mean(vals):.3f}, "
                    f"median={stats.median(vals):.3f}, stdev={(stats.pstdev(vals)):.3f}"
                )
        return self.reply("\n".join(msg))

    # ---- Complaint (slot filling) ---- #
    def handle_complaint(self, text: str) -> str:
        # Start or continue a complaint ticket
        if not self.pending_task or self.pending_task.get("type") != "complaint":
            self.pending_task = {
                "type": "complaint",
                "slots": {"product": None, "issue": None, "order_id": None},
            }
            return self.reply("I'm opening a support ticket. What product is this about?")
        # continue
        slots = self.pending_task["slots"]
        if slots["product"] is None:
            slots["product"] = text
            return self.reply("Got it. Briefly describe the issue.")
        if slots["issue"] is None:
            slots["issue"] = text
            return self.reply("Please share your order ID (or say 'none').")
        if slots["order_id"] is None:
            oid = text.strip()
            slots["order_id"] = None if oid.lower() == 'none' else oid
            ticket_id = f"TKT-{random.randint(10000, 99999)}"
            self.pending_task = None
            return self.reply(
                f"Thanks! Ticket {ticket_id} created. Product: {slots['product']}. "
                f"Issue: {slots['issue']}. Order ID: {slots['order_id'] or 'N/A'}. Our team will follow up."
            )
        return self.reply("Your ticket is already created. Anything else?")

    def handle_smalltalk(self, text: str) -> str:
        s = sentiment_score(text)
        if s > 0.02:
            return self.reply(random.choice(SMALLTALK) + " 😊")
        if s < -0.02:
            return self.reply("Sorry to hear that. Tell me more—maybe I can help.")
        return self.reply(random.choice(SMALLTALK))

    # --------------- Router --------------- #
    def handle(self, user_text: str) -> str:
        user_text_raw = user_text
        user_text = normalize(user_text)
        self.mem.add_history("user", user_text_raw)

        # If we are in a slot-filling flow, prioritize it unless commands
        if self.pending_task and user_text.startswith('/') is False:
            if self.pending_task.get("type") == "complaint":
                return self.handle_complaint(user_text_raw)

        intent, m = detect_intent(user_text)
        if intent == "greet":
            return self.handle_greet()
        if intent == "goodbye":
            return self.handle_goodbye()
        if intent == "set_name" and m:
            return self.handle_set_name(m)
        if intent == "ask_time":
            return self.handle_time()
        if intent == "ask_date":
            return self.handle_date()
        if intent == "calculator":
            return self.handle_calculator(user_text_raw)
        if intent == "faq":
            return self.handle_faq(user_text)
        if intent == "load_csv" and m:
            return self.handle_load_csv(m)
        if intent == "setname_cmd" and m:
            return self.handle_setname_cmd(m)
        if intent == "help":
            return self.handle_help()
        if intent == "reset":
            return self.handle_reset()
        if intent == "export":
            return self.handle_export()
        if intent == "summary":
            return self.handle_summary()
        if intent == "complaint":
            return self.handle_complaint(user_text_raw)
        # default
        return self.handle_smalltalk(user_text_raw)


# --------------------------- Entry Point ---------------------------- #

def banner():
    print("=" * 68)
    print("Big AI Chatbot (No External Libraries)")
    print("Type /help for commands. Type 'bye' to exit.")
    print("=" * 68)


def run():
    bot = Chatbot()
    banner()
    # greet on start
    print(bot.handle_greet())
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            user = "bye"
        if not user.strip():
            continue
        resp = bot.handle(user)
        print(resp)
        if resp in GOODBYES:
            break


if __name__ == "__main__":
    run()
