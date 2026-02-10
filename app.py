import os
import re
import json
import time
import sqlite3
from datetime import datetime
from io import StringIO, BytesIO
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytesseract
from groq import Groq
import spacy
from dateutil import parser

nlp = spacy.load("en_core_web_sm")

# ========================= CURRENCY HANDLING (INR & USD ONLY) =========================
CURRENCY_SYMBOLS = {
    'INR': '₹',
    'USD': '$'
}
# Exchange rates dictionary (keeping for backward compatibility in analytics)
EXCHANGE_RATES = {
    'INR': 1.0,
    'USD': 83.0
}

# USD to INR exchange rate
EXCHANGE_RATE_USD_TO_INR = 83.0

def detect_currency(text):
    """Detect currency from text - INR or USD only"""
    # Check for USD indicators
    usd_patterns = [r'\$', r'\bUSD\b', r'\bDollar']
    for pattern in usd_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'USD'
    
    # Default to INR (includes ₹, Rs, Rupee, or no currency symbol)
    return 'INR'

def convert_to_inr(amount, currency):
    """Convert amount to INR - only USD conversion needed"""
    if currency == 'USD':
        return round(amount * EXCHANGE_RATE_USD_TO_INR, 2)
    # INR or any other currency stays as-is
    return round(amount, 2)

# ========================= INTELLIGENT CATEGORIZATION (ENHANCED) =========================
CATEGORY_KEYWORDS = {
    'Food': [
        # Restaurants & Cafes
        'restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'taco', 'sushi',
        'bar', 'grill', 'kitchen', 'bistro', 'diner', 'eatery', 'food',
        'lunch', 'dinner', 'breakfast', 'meal', 'cuisine',
        # Delivery & Fast Food
        'doordash', 'ubereats', 'zomato', 'swiggy', 'dominos', 'mcdonalds', 'kfc',
        'subway', 'starbucks', 'dunkin', 'popeyes', 'chipotle',
        # Food Items
        'roll', 'pasta', 'noodle', 'rice', 'chicken', 'fish', 'meat', 'salad',
        'sandwich', 'wrap', 'bowl', 'curry', 'biryani', 'tikka', 'masala',
        'fries', 'wings', 'nugget', 'taco', 'burrito', 'quesadilla',
        'ramen', 'pho', 'sushi', 'tempura', 'teriyaki',
        # Beverages
        'latte', 'cappuccino', 'espresso', 'mocha', 'smoothie', 'shake', 'juice',
        'tea', 'chai', 'lassi', 'soda', 'beer', 'wine',
        # Merchant Names
        'pokeater', 'poke', 'tiant', 'city taste', 'trattor', 'naanc',
        'taste', 'flavor', 'spice', 'cuisine', 'house'
    ],
    'Grocery': [
        'grocery', 'supermarket', 'market', 'walmart', 'target', 'costco',
        'kroger', 'safeway', 'whole foods', 'trader joe', 'aldi', 'fresh',
        'produce', 'mart', 'store', 'big basket', 'grofers', 'blinkit'
    ],
    'Shopping': [
        'retail', 'shop', 'mall', 'boutique', 'amazon', 'ebay',
        'clothing', 'fashion', 'accessories', 'shoes', 'apparel', 'nike',
        'adidas', 'zara', 'h&m', 'uniqlo', 'electronics', 'best buy',
        'flipkart', 'myntra', 'ajio'
    ],
    'Medical': [
        'pharmacy', 'medical', 'hospital', 'clinic', 'doctor', 'health',
        'dental', 'care', 'medicine', 'drug', 'cvs', 'walgreens', 'rite aid',
        'apollo', 'medplus', 'netmeds', '1mg', 'pharmac'
    ],
    'Travel': [
        'hotel', 'flight', 'airline', 'travel', 'uber', 'lyft', 'taxi',
        'cab', 'transport', 'gas', 'fuel', 'petrol', 'parking', 'toll',
        'airbnb', 'booking', 'ola', 'rapido', 'train', 'bus',
        'perrier', 'sparkling water'  # Travel beverages
    ],
    'Entertainment': [
        'cinema', 'movie', 'theater', 'ticket', 'game', 'sport', 'gym',
        'fitness', 'concert', 'event', 'park', 'museum', 'netflix',
        'spotify', 'gaming', 'entertainment', 'pvr', 'inox'
    ]
}

def categorize_receipt(merchant, items_text):
    """Intelligently categorize receipt - ENHANCED with better scoring"""
    combined_text = f"{merchant} {items_text}".lower()
    
    category_scores = {category: 0 for category in CATEGORY_KEYWORDS}
    
    # Score each category based on keyword matches
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            # Count all occurrences (not just presence)
            count = combined_text.count(keyword)
            if count > 0:
                # Food category gets bonus weight for food items
                if category == 'Food' and any(food_item in keyword for food_item in 
                    ['roll', 'pasta', 'rice', 'chicken', 'salad', 'curry', 'tikka', 
                     'fries', 'wings', 'burger', 'pizza', 'sandwich', 'latte']):
                    category_scores[category] += count * 2  # Double weight for food items
                else:
                    category_scores[category] += count
    
    # Get category with highest score
    max_category = max(category_scores.items(), key=lambda x: x[1])
    
    # Only assign category if confidence is high enough
    if max_category[1] > 0:
        return max_category[0]
    
    return 'Others'
# ========================= VALIDATION =========================
def validate_receipt_row(row, full_df):
    validations = []

    tip_value = row.get("tip", 0) or 0
    total_ok = abs((row["subtotal"] + row["tax"] + tip_value) - row["total"]) <= 1
    
    if tip_value > 0:
        validations.append((
            "Total Validation",
            f"Subtotal ({row['subtotal']}) + Tax ({row['tax']}) + Tip ({tip_value}) ≈ Total ({row['total']})",
            total_ok
        ))
    else:
        validations.append((
            "Total Validation",
            f"Subtotal ({row['subtotal']}) + Tax ({row['tax']}) ≈ Total ({row['total']})",
            total_ok
        ))

    dup = full_df[
        (full_df["merchant"] == row["merchant"]) &
        (full_df["date"] == row["date"]) &
        (full_df["total"] == row["total"]) &
        (full_df["id"] != row["id"])
    ]
    validations.append((
        "Duplicate Detection",
        "No duplicate found" if dup.empty else "Duplicate receipt detected",
        dup.empty
    ))

    if row["subtotal"] > 0 and row["tax"] > 0:
        rate = round((row["tax"] / row["subtotal"]) * 100, 2)
        validations.append((
            "Tax Rate Validation",
            f"Expected ~0–30%, Actual: {rate}%",
            0 <= rate <= 30
        ))
    else:
        validations.append((
            "Tax Rate Validation",
            "Tax or Subtotal is zero" if row["subtotal"] > 0 else "Subtotal missing",
            True
        ))

    date_str = str(row["date"]).strip()
    date_patterns = [
        r"\b\d{2}/\d{2}/\d{2}\b", r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b", r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
        r"\b[A-Za-z]+\s+\d{1,2},?\s*\d{4}?\b"
    ]
    date_ok = any(re.search(p, date_str, re.IGNORECASE) for p in date_patterns) or len(date_str) >= 8

    validations.append((
        "Date Validation",
        "Valid date format detected" if date_ok else "Date missing or unrecognized format",
        date_ok
    ))

    required_ok = bool(row["merchant"] and row["total"] > 0)
    validations.append((
        "Required Fields",
        "All required fields present" if required_ok else "Missing required fields",
        required_ok
    ))

    return validations

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(page_title="Receipt & Invoice Digitizer", page_icon="🧾", layout="wide")

DB_FILE = "receipts.db"

CUSTOM_CSS = """
<style>
.main { background: radial-gradient(circle at top left, #0b1220 0%, #05070f 70%); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
.big-title{
    font-size: 34px; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0a1222 0%, #05070f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.card{
    border-radius: 18px; padding: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 12px;
}
.stButton>button{
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 16px !important;
}
.category-badge{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}
.cat-food{ background: #34d399; color: #064e3b; }
.cat-grocery{ background: #60a5fa; color: #1e3a8a; }
.cat-shopping{ background: #a78bfa; color: #4c1d95; }
.cat-medical{ background: #f87171; color: #7f1d1d; }
.cat-travel{ background: #fbbf24; color: #78350f; }
.cat-entertainment{ background: #fb923c; color: #7c2d12; }
.cat-others{ background: #94a3b8; color: #1e293b; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ==========================================
# DATABASE FUNCTIONS - ENHANCED
# ==========================================
def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            merchant TEXT,
            date TEXT,
            time TEXT,
            payment TEXT,
            category TEXT,
            currency TEXT,
            subtotal REAL,
            tax REAL,
            tip REAL,
            total REAL,
            total_inr REAL,
            items_json TEXT
        )
    """)
    
    cur.execute("PRAGMA table_info(receipts)")
    columns = [col[1] for col in cur.fetchall()]
    
    columns_to_add = {
        "time": "TEXT DEFAULT ''",
        "tip": "REAL DEFAULT 0",
        "category": "TEXT DEFAULT 'Others'",
        "currency": "TEXT DEFAULT 'INR'",  # Default to INR
        "total_inr": "REAL DEFAULT 0"
    }
    
    for col_name, col_def in columns_to_add.items():
        if col_name not in columns:
            try:
                cur.execute(f"ALTER TABLE receipts ADD COLUMN {col_name} {col_def}")
                con.commit()
            except Exception as e:
                pass
    
    cur.execute("DROP TABLE IF EXISTS receipts_tmp")
    con.commit()
    con.close()

def db_load_all():
    con = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM receipts ORDER BY id ASC", con)
    con.close()
    return df

def db_insert_receipt(data: dict):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO receipts (id, created_at, merchant, date, time, payment, category, currency,
                             subtotal, tax, tip, total, total_inr, items_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["id"],
        data["created_at"],
        data.get("merchant", ""),
        data.get("date", ""),
        data.get("time", ""),
        data.get("payment", "CASH"),
        data.get("category", "Others"),
        data.get("currency", "INR"),
        float(data.get("subtotal") or 0),
        float(data.get("tax") or 0),
        float(data.get("tip") or 0),
        float(data.get("total") or 0),
        float(data.get("total_inr") or 0),
        data.get("items_json", "[]")
    ))
    con.commit()
    con.close()

def db_clear_all():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("DELETE FROM receipts")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='receipts'")
    con.commit()
    con.close()

def db_get_next_id():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT MAX(id) FROM receipts")
    maxid = cur.fetchone()[0]
    con.close()
    return 1 if maxid is None else int(maxid) + 1

def db_renumber_ids():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(receipts)")
    columns = [col[1] for col in cur.fetchall()]
    
    cur.execute("DROP TABLE IF EXISTS receipts_tmp")
    cur.execute("""
        CREATE TABLE receipts_tmp (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            merchant TEXT,
            date TEXT,
            time TEXT,
            payment TEXT,
            category TEXT,
            currency TEXT,
            subtotal REAL,
            tax REAL,
            tip REAL,
            total REAL,
            total_inr REAL,
            items_json TEXT
        )
    """)

    select_cols = ["created_at", "merchant", "date", "time", "payment", "category", 
                   "currency", "subtotal", "tax", "tip", "total", "total_inr", "items_json"]
    
    available_cols = [c for c in select_cols if c in columns]
    
    cur.execute(f"SELECT {', '.join(available_cols)} FROM receipts ORDER BY id ASC")
    rows = cur.fetchall()

    for new_id, row in enumerate(rows, start=1):
        row_dict = dict(zip(available_cols, row))
        
        cur.execute("""
            INSERT INTO receipts_tmp (id, created_at, merchant, date, time, payment, category,
                                     currency, subtotal, tax, tip, total, total_inr, items_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_id,
            row_dict.get("created_at", ""),
            row_dict.get("merchant", ""),
            row_dict.get("date", ""),
            row_dict.get("time", ""),
            row_dict.get("payment", "CASH"),
            row_dict.get("category", "Others"),
            row_dict.get("currency", "INR"),
            row_dict.get("subtotal", 0),
            row_dict.get("tax", 0),
            row_dict.get("tip", 0),
            row_dict.get("total", 0),
            row_dict.get("total_inr", 0),
            row_dict.get("items_json", "[]")
        ))

    cur.execute("DELETE FROM receipts")
    cur.execute("INSERT INTO receipts SELECT * FROM receipts_tmp ORDER BY id ASC")
    cur.execute("DROP TABLE IF EXISTS receipts_tmp")
    con.commit()
    con.close()

def db_delete_by_ids(ids):
    if not ids:
        return
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(ids))
    cur.execute(f"DELETE FROM receipts WHERE id IN ({placeholders})", ids)
    con.commit()
    con.close()
    db_renumber_ids()

init_db()
# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def pdf_to_images(uploaded_pdf):
    poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
    return convert_from_bytes(uploaded_pdf.read(), dpi=250, poppler_path=poppler_path)

def safe_float(x):
    try:
        if x is None:
            return 0.0
        if isinstance(x, str) and x.strip() == "":
            return 0.0
        return float(x)
    except:
        return 0.0

def parse_price_to_float(val):
    if val is None:
        return 0.0
    s = str(val).strip().replace(",", "")
    s = re.sub(r'[₹$€£RM]', '', s)
    s = re.sub(r"[^0-9.]", "", s)
    try:
        return float(s) if s else 0.0
    except:
        return 0.0

def clean_line_items(items_df: pd.DataFrame):
    if items_df is None or len(items_df) == 0:
        return items_df

    bad_keywords = [
        "subtotal", "total", "tax", "balance", "change",
        "amount", "payment", "cash", "card", "visa", "tip", "gratuity",
        "service", "fee", "discount", "coupon"
    ]

    cleaned_rows = []
    for _, row in items_df.iterrows():
        name = str(row.get("name", "")).strip().lower()

        if any(bk in name for bk in bad_keywords):
            continue

        price = parse_price_to_float(row.get("price", 0))
        qty = parse_price_to_float(row.get("qty", 1))

        if price == 0:
            continue

        cleaned_rows.append({
            "name": row.get("name", "").strip(),
            "qty": int(qty if qty > 0 else 1),
            "price": price
        })

    return pd.DataFrame(cleaned_rows)

def calculate_subtotal_from_items(items_df: pd.DataFrame):
    """Calculate subtotal from line items - MOST ACCURATE METHOD"""
    if items_df is None or len(items_df) == 0:
        return 0.0
    
    subtotal = 0.0
    for _, row in items_df.iterrows():
        qty = parse_price_to_float(row.get("qty", 1))
        price = parse_price_to_float(row.get("price", 0))
        if qty <= 0:
            qty = 1
        subtotal += qty * price
    
    return round(subtotal, 2)

def estimate_missing_tax(subtotal, total, tip):
    """Estimate tax if missing but total > subtotal"""
    if subtotal > 0 and total > subtotal:
        derived_tax = total - subtotal - tip
        tax_rate = (derived_tax / subtotal) * 100
        
        if 0 <= tax_rate <= 30:
            return round(derived_tax, 2)
    
    return 0.0

def validate_and_correct_amounts(data, items_df):
    """CRITICAL FIX: Always use TOTAL - TAX = SUBTOTAL formula"""
    tax = data.get("tax", 0)
    tip = data.get("tip", 0)
    total = data.get("total", 0)
    
    # PRIORITY 1: If we have total and tax, calculate subtotal
    if total > 0 and tax >= 0:
        # Formula: Subtotal = Total - Tax - Tip
        calculated_subtotal = total - tax - tip
        
        if calculated_subtotal > 0:
            data["subtotal"] = round(calculated_subtotal, 2)
        else:
            # Fallback: Try to calculate from items
            items_subtotal = calculate_subtotal_from_items(items_df)
            if items_subtotal > 0:
                data["subtotal"] = items_subtotal
            else:
                data["subtotal"] = 0
    
    # PRIORITY 2: If we don't have tax but have total, try to derive it
    elif total > 0 and tax == 0:
        # Try to calculate subtotal from items
        items_subtotal = calculate_subtotal_from_items(items_df)
        
        if items_subtotal > 0:
            # Derive tax from: Tax = Total - Subtotal - Tip
            derived_tax = total - items_subtotal - tip
            
            if 0 <= derived_tax <= items_subtotal * 0.3:  # Tax should be 0-30% of subtotal
                data["subtotal"] = items_subtotal
                data["tax"] = round(derived_tax, 2)
            else:
                # Tax doesn't make sense, just use formula
                data["subtotal"] = round(total - tip, 2)
                data["tax"] = 0
        else:
            # No items to calculate from
            data["subtotal"] = round(total - tip, 2)
            data["tax"] = 0
    
    # PRIORITY 3: If we only have subtotal and tax, calculate total
    elif data.get("subtotal", 0) > 0 and total == 0:
        subtotal = data.get("subtotal", 0)
        data["total"] = round(subtotal + tax + tip, 2)
    
    # Ensure all values are non-negative
    data["subtotal"] = max(0, data.get("subtotal", 0))
    data["tax"] = max(0, data.get("tax", 0))
    data["tip"] = max(0, data.get("tip", 0))
    data["total"] = max(0, data.get("total", 0))
    
    return data

def normalize_date(date_str):
    try:
        dt = parser.parse(date_str, fuzzy=True, dayfirst=True)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except:
        return "", "00:00:00"

def normalize_money(val):
    return round(parse_price_to_float(val), 2)
# ==========================================
# ENHANCED IMAGE PREPROCESSING
# ==========================================
def make_clean_image(image: Image.Image):
    """Method 1: CLAHE + Adaptive Threshold"""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    clean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return clean

def make_clean_image_v2(image: Image.Image):
    """Method 2: Bilateral + Otsu's Threshold"""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def enhance_image_for_ocr(image: Image.Image):
    """Method 3: Sharpening + Contrast Enhancement"""
    img_array = np.array(image.convert("RGB"))
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img_array, -1, kernel_sharpen)
    
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced)

# ==========================================
# MULTI-PASS OCR ENGINE
# ==========================================
def setup_tesseract():
    tpath = st.secrets.get("TESSERACT_PATH", "")
    if tpath and os.path.exists(tpath):
        pytesseract.pytesseract.tesseract_cmd = tpath

def ocr_extract_text(pil_img: Image.Image):
    """Enhanced OCR with multiple PSM modes"""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 3",
    ]
    
    texts = []
    for config in configs:
        try:
            text = pytesseract.image_to_string(gray, config=config)
            texts.append(text.strip())
        except:
            continue
    
    return max(texts, key=len) if texts else ""

setup_tesseract()

# ==========================================
# SMART MERCHANT EXTRACTION
# ==========================================
def correct_merchant_name(name):
    """Fix common OCR errors in merchant names"""
    if not name:
        return name
    
    corrections = {
        '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
    }
    
    corrected = []
    for i, char in enumerate(name):
        if char.isdigit() and i > 0 and i < len(name) - 1:
            if name[i-1].isalpha() or (i < len(name) - 1 and name[i+1].isalpha()):
                corrected.append(corrections.get(char, char))
            else:
                corrected.append(char)
        else:
            corrected.append(char)
    
    result = ''.join(corrected)
    
    known_patterns = {
        'P0KEATER': 'POKEATER',
        'POK EATER': 'POKEATER',
        'T1 ANT': 'TIANT',
        'T1ANT': 'TIANT',
        'PHARMAC0': 'PHARMACO',
        'SAN RETAI': 'SAN RETAIL',
    }
    
    for pattern, replacement in known_patterns.items():
        if pattern in result.upper():
            result = result.upper().replace(pattern, replacement)
    
    return result

def extract_merchant_smart(text):
    """Multi-method merchant extraction"""
    lines = text.strip().split('\n')
    merchant_candidates = []
    
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 2 and not any(keyword in line.lower() for keyword in 
                                    ['receipt', 'invoice', 'bill', 'date', 'time', 'total', 'tax', 'subtotal']):
            merchant_candidates.append(line)
    
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "ORG":
            merchant_candidates.append(ent.text)
    
    merchant_patterns = [
        r'(?:merchant|store|shop)[\s:]+([A-Z][A-Za-z0-9\s]+)',
        r'^([A-Z][A-Z\s]{3,20})$',
    ]
    
    for pattern in merchant_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        merchant_candidates.extend(matches)
    
    if merchant_candidates:
        best = min(merchant_candidates, key=lambda x: len(x) if len(x) < 30 else 100)
        return correct_merchant_name(best)
    
    return ""
# ==========================================
# REGEX FALLBACK EXTRACTION
# ==========================================
def regex_fallback(text):
    """Enhanced regex with multiple patterns"""
    data = {}
    
    total_patterns = [
        r"(total|grand total|amount due|total due)[\s:]*[\$₹€£RM]?\s*(\d+[\.,]\d{2})",
        r"(total)[\s:]*(\d+[\.,]\d{2})",
    ]
    
    tax_patterns = [
        r"(tax|gst|vat|sales tax)[\s:]*[\$₹€£RM]?\s*(\d+[\.,]\d{2})",
        r"(tax)[\s:]*(\d+[\.,]\d{2})",
    ]
    
    subtotal_patterns = [
        r"(subtotal|sub total|sub-total|items)[\s:]*[\$₹€£RM]?\s*(\d+[\.,]\d{2})",
        r"(subtotal)[\s:]*(\d+[\.,]\d{2})",
    ]
    
    tip_patterns = [
        r"(tip|gratuity)[\s:]*[\$₹€£RM]?\s*(\d+[\.,]\d{2})",
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["total"] = normalize_money(match.group(2))
            break
    
    for pattern in tax_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["tax"] = normalize_money(match.group(2))
            break
    
    for pattern in subtotal_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["subtotal"] = normalize_money(match.group(2))
            break
    
    for pattern in tip_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["tip"] = normalize_money(match.group(2))
            break
    
    # Date extraction with multiple formats
    date_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            try:
                d, t = normalize_date(match.group(0))
                data["date"] = d
                data["time"] = t
                break
            except:
                continue
    
    # Time extraction
    time_match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b", text)
    if time_match:
        try:
            time_str = time_match.group(0)
            time_obj = parser.parse(time_str, fuzzy=True)
            data["time"] = time_obj.strftime("%H:%M:%S")
        except:
            pass

    return data

# ==========================================
# GROQ EXTRACTION - ENHANCED PROMPT
# ==========================================
GROQ_SCHEMA_PROMPT = """
You are an EXPERT receipt/invoice parser. Extract data with MAXIMUM ACCURACY.

Return STRICT JSON ONLY.

JSON schema:
{
  "merchant": "",
  "date": "",
  "time": "",
  "payment": "CASH or CARD or UPI or OTHER",
  "subtotal": 0,
  "tax": 0,
  "tip": 0,
  "total": 0,
  "items": [
    {"name": "", "qty": 1, "price": 0}
  ]
}

CRITICAL EXTRACTION RULES:
1. MERCHANT: Extract business name from TOP of receipt. NOT address/phone.
2. NUMBERS: Preserve decimals EXACTLY. 229.86 means 229.86 NOT 229.66.
3. SUBTOTAL: MUST equal sum of all item prices. Verify!
4. ITEMS: Extract ONLY products. Skip "subtotal", "tax", "total", "service fee".
5. DATE: Format as YYYY-MM-DD, DD-MM-YYYY, or MM/DD/YYYY.
6. TIME: Format as HH:MM or HH:MM:SS. Look near date or at top.
7. TIP: Look for "tip" or "gratuity". If missing → 0.
8. TAX: Look for "tax", "gst", "vat". If missing → 0.
9. TOTAL: Should equal Subtotal + Tax + Tip.
10. PAYMENT: Card numbers/"VISA"/"MASTERCARD" → CARD. Otherwise → CASH.

VALIDATION BEFORE RETURNING:
- Total = Subtotal + Tax + Tip ✓
- Subtotal = Sum of (item.qty × item.price) ✓
- All numbers preserved exactly ✓

Return ONLY the JSON object. No markdown, no explanations.
"""

def groq_validate_key(api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        return False

    client = Groq(api_key=api_key)
    models_to_try = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    for model_name in models_to_try:
        try:
            _ = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=5,
                temperature=0
            )
            return True
        except Exception:
            continue
    return False
# ==========================================
# MAIN RECEIPT PROCESSING PIPELINE
# ==========================================
def process_receipt_pipeline(image, api_key):
    """Complete extraction pipeline with all enhancements"""
    client = Groq(api_key=api_key)

    # ============ STEP 1: MULTI-PASS OCR ============
    enhanced_img = enhance_image_for_ocr(image)
    ocr_enhanced = ocr_extract_text(enhanced_img)
    
    cleaned_v1 = make_clean_image(image)
    cleaned_v2 = make_clean_image_v2(image)
    
    ocr_clean1 = ocr_extract_text(Image.fromarray(cleaned_v1))
    ocr_clean2 = ocr_extract_text(Image.fromarray(cleaned_v2))
    
    # Combine all OCR results
    combined_text = f"{ocr_enhanced}\n{ocr_clean1}\n{ocr_clean2}"
    
    # Detect currency from text
    detected_currency = detect_currency(combined_text)

    # ============ STEP 2: GROQ LLM EXTRACTION ============
    prompt = f"{GROQ_SCHEMA_PROMPT}\n\nOCR TEXT:\n{combined_text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"Extract receipt data in JSON. Preserve number accuracy."},
                {"role":"user","content":prompt}
            ],
            temperature=0.05,
            max_tokens=1500
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","")
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception as e:
        data = {}

    # ============ STEP 3: REGEX FALLBACK ============
    fallback = regex_fallback(combined_text)
    
    for key, value in fallback.items():
        if not data.get(key) or data.get(key) == 0 or data.get(key) == "":
            data[key] = value

    # ============ STEP 4: SMART MERCHANT EXTRACTION ============
    if not data.get("merchant"):
        data["merchant"] = extract_merchant_smart(combined_text)
    else:
        data["merchant"] = correct_merchant_name(data["merchant"])

    # ============ STEP 5: DATE/TIME NORMALIZATION ============
    raw_date = data.get("date", "")
    raw_time = data.get("time", "")
    
    if raw_date:
        try:
            if raw_time and ":" in str(raw_time):
                dt = parser.parse(raw_date, fuzzy=True, dayfirst=True)
                date_iso = dt.strftime("%Y-%m-%d")
                time_obj = parser.parse(raw_time, fuzzy=True)
                time_iso = time_obj.strftime("%H:%M:%S")
            else:
                date_iso, time_iso = normalize_date(raw_date)
        except:
            date_iso, time_iso = normalize_date(raw_date)
    else:
        date_iso, time_iso = "", "00:00:00"
    
    data["date"] = date_iso
    data["time"] = time_iso
    
    # ============ STEP 6: MONEY NORMALIZATION ============
    data["subtotal"] = normalize_money(data.get("subtotal"))
    data["tax"] = normalize_money(data.get("tax"))
    data["tip"] = normalize_money(data.get("tip"))
    data["total"] = normalize_money(data.get("total"))
    
    if not data.get("payment") or str(data.get("payment")).strip() == "":
        data["payment"] = "CASH"

    # ============ STEP 7: LINE ITEMS PROCESSING ============
    items = data.get("items", [])
    items_df = pd.DataFrame(items) if items else pd.DataFrame(columns=["name","qty","price"])
    items_df = clean_line_items(items_df)
    
    # ============ STEP 8: INTELLIGENT VALIDATION & CORRECTION ============
    data = validate_and_correct_amounts(data, items_df)
    
    # ============ STEP 9: CURRENCY CONVERSION ============
    total_inr = convert_to_inr(data["total"], detected_currency)
    
    # ============ STEP 10: INTELLIGENT CATEGORIZATION ============
    items_text = " ".join([row["name"] for _, row in items_df.iterrows()])
    category = categorize_receipt(data["merchant"], items_text)
    
    return {
        "merchant": data.get("merchant",""),
        "date": data.get("date",""),
        "time": data.get("time",""),
        "payment": data.get("payment","CASH"),
        "category": category,
        "currency": detected_currency,
        "subtotal": data.get("subtotal",0),
        "tax": data.get("tax",0),
        "tip": data.get("tip",0),
        "total": data.get("total",0),
        "total_inr": total_inr,
        "items_json": items_df.to_json(orient="records"),
        "clean_image": cleaned_v1,
        "ocr_text": combined_text
    }
# ==========================================
# SIDEBAR AUTHENTICATION
# ==========================================
st.sidebar.markdown("## 🔐 Authentication")

groq_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()

if not groq_key:
    st.warning("🔑 Please add GROQ_API_KEY in `.streamlit/secrets.toml` to continue.")
    st.stop()

with st.sidebar:
    st.info("Validating Groq API key...")

if not groq_validate_key(groq_key):
    st.sidebar.error("❌ Invalid Groq API Key OR quota issue.")
    st.stop()

st.sidebar.success("✅ API Key Verified")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Records"):
    db_clear_all()
    st.sidebar.success("All records cleared ✅")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("✅ Database: SQLite (receipts.db)\n\n✅ Multi-currency support\n\n✅ AI Categorization")

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="big-title">🧾 Receipt & Invoice Digitizer</div>', unsafe_allow_html=True)
st.caption("Advanced OCR + AI extraction • Multi-currency support • Intelligent categorization • Comprehensive analytics")

tabs = st.tabs(["📥 Vault & Upload", "🕘 History", "📊 Analytics Dashboard"])

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None

if "processed_outputs" not in st.session_state:
    st.session_state.processed_outputs = {}
# ==========================================
# TAB 1: UPLOAD & PROCESSING
# ==========================================
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Upload Document(s)")
    st.caption("Supports JPG, PNG, PDF • Multi-currency • Auto-categorization")

    uploaded_files = st.file_uploader(
        "Upload Receipt(s)",
        type=["jpg", "png", "pdf", "jpeg"],
        accept_multiple_files=True,
        key=f"multi_receipt_uploader_{st.session_state.uploader_key}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files is not None and len(uploaded_files) > 0:
        current_signature = tuple(sorted([f"{f.name}_{f.size}" for f in uploaded_files]))

        if st.session_state.last_upload_signature is not None and current_signature != st.session_state.last_upload_signature:
            st.session_state.processed_outputs = {}
            st.session_state.uploader_key += 1
            st.session_state.last_upload_signature = None
            st.rerun()
        else:
            st.session_state.last_upload_signature = current_signature

    if uploaded_files:
        tasks = []
        for uploaded in uploaded_files:
            if uploaded.type == "application/pdf":
                pdf_pages = pdf_to_images(uploaded)
                for pno, pimg in enumerate(pdf_pages, start=1):
                    tasks.append((f"{uploaded.name} (Page {pno})", pimg))
            else:
                tasks.append((uploaded.name, Image.open(uploaded)))

        total_tasks = len(tasks)

        # FIXED: Only show Process All section if multiple files uploaded
        if total_tasks > 1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ⚡ Actions")
            colx1, colx2 = st.columns([1, 2])
            with colx1:
                process_all = st.button("✅ Process All")
            with colx2:
                st.info(f"📌 Total queued: **{total_tasks}** receipts")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            process_all = False  # Don't process all if single file

        progress_bar = st.progress(0)
        status_text = st.empty()

        def run_processing(image):
            result = process_receipt_pipeline(image, groq_key)

            payload = {
                "id": db_get_next_id(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "merchant": result["merchant"],
                "date": result["date"],
                "time": result["time"],
                "payment": result["payment"],
                "category": result["category"],
                "currency": result["currency"],
                "subtotal": result["subtotal"],
                "tax": result["tax"],
                "tip": result["tip"],
                "total": result["total"],
                "total_inr": result["total_inr"],
                "items_json": result["items_json"],
            }
            return payload, result

        if process_all and total_tasks > 1:  # FIXED: Only run if multiple files
            done = 0
            cooldown_seconds = 1
            wait_box = st.empty()

            for i, (title, img) in enumerate(tasks, start=1):
                status_text.info(f"Processing: **{title}** ({i}/{total_tasks})")
                with st.spinner(f"Extracting from {title}..."):
                    payload, result = run_processing(img)
                    st.session_state.processed_outputs[title] = {
                        "clean_image": result["clean_image"],
                        "data": payload
                    }

                done += 1
                progress_bar.progress(int(done * 100 / total_tasks))

                if i != total_tasks:
                    for sec in range(cooldown_seconds, 0, -1):
                        wait_box.warning(f"⏳ Cooling down... {sec}s")
                        time.sleep(1)
                    wait_box.empty()

            status_text.success(f"✅ Processed {total_tasks} receipts!")
            st.toast("✅ Process All completed!", icon="✅")
            st.rerun()

        for idx, (file_title, image) in enumerate(tasks, start=1):
            st.markdown("---")
            st.markdown(f"## 📄 {idx}. `{file_title}`")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Original Receipt")
                st.image(image, width=360)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🧹 Cleaned / Processed")
                if file_title in st.session_state.processed_outputs:
                    clean_img = st.session_state.processed_outputs[file_title]["clean_image"]
                    st.image(clean_img, width=360)
                else:
                    st.info("Not processed yet.")
                st.markdown("</div>", unsafe_allow_html=True)

            process_key = f"process_{file_title}_{idx}"
            if st.button(f"🔍 Preprocess ({file_title})", key=process_key):
                status_text.info(f"Processing: **{file_title}**")

                with st.spinner("Running AI Extraction..."):
                    result = process_receipt_pipeline(image, groq_key)

                st.session_state.processed_outputs[file_title] = {
                    "clean_image": result["clean_image"],
                    "data": {
                        "merchant": result["merchant"],
                        "date": result["date"],
                        "time": result["time"],
                        "payment": result["payment"],
                        "category": result["category"],
                        "currency": result["currency"],
                        "subtotal": result["subtotal"],
                        "tax": result["tax"],
                        "tip": result["tip"],
                        "total": result["total"],
                        "total_inr": result["total_inr"],
                        "items_json": result["items_json"]
                    }
                }

                progress_bar.progress(100)
                status_text.success("✅ Preprocessing complete!")
                st.rerun()

            if file_title in st.session_state.processed_outputs:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🧪 Receipt Validation")
                
                extracted = st.session_state.processed_outputs[file_title]["data"]
                
                # Show extracted info
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Category", extracted["category"])
                with info_col2:
                    currency_symbol = CURRENCY_SYMBOLS.get(extracted["currency"], extracted["currency"])
                    st.metric("Currency", f"{currency_symbol} ({extracted['currency']})")
                with info_col3:
                    st.metric("Total (INR)", f"₹ {extracted['total_inr']:.2f}")

                if st.button(f"✅ Validate Receipt ({file_title})", key=f"validate_{file_title}"):
                    temp_row = {
                        "id": -1,
                        "merchant": extracted["merchant"],
                        "date": extracted["date"],
                        "subtotal": extracted["subtotal"],
                        "tax": extracted["tax"],
                        "tip": extracted.get("tip", 0),
                        "total": extracted["total"]
                    }

                    df_existing = db_load_all()
                    validations = validate_receipt_row(temp_row, df_existing)

                    all_ok = True
                    for title, message, status in validations:
                        if status:
                            st.success(f"✔ {title} — {message}")
                        else:
                            st.error(f"✖ {title} — {message}")
                            all_ok = False
                    st.session_state[f"validation_passed_{file_title}"] = all_ok

                st.markdown("</div>", unsafe_allow_html=True)

            if file_title in st.session_state.processed_outputs:
                if st.session_state.get(f"validation_passed_{file_title}", False):
                    if st.button(f"💾 Save Receipt ({file_title})", key=f"save_{file_title}"):
                        extracted = st.session_state.processed_outputs[file_title]["data"]

                        payload = {
                            "id": db_get_next_id(),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "merchant": extracted["merchant"],
                            "date": extracted["date"],
                            "time": extracted["time"],
                            "payment": extracted["payment"],
                            "category": extracted["category"],
                            "currency": extracted["currency"],
                            "subtotal": extracted["subtotal"],
                            "tax": extracted["tax"],
                            "tip": extracted.get("tip", 0),
                            "total": extracted["total"],
                            "total_inr": extracted["total_inr"],
                            "items_json": extracted["items_json"],
                        }

                        db_insert_receipt(payload)
                        st.success(f"✅ Receipt saved! ID: {payload['id']} | Category: {payload['category']}")

                        del st.session_state.processed_outputs[file_title]
                        if f"validation_passed_{file_title}" in st.session_state:
                            del st.session_state[f"validation_passed_{file_title}"]
                        st.rerun()
# ==========================================
# TAB 2: HISTORY WITH CATEGORIES (FIXED)
# ==========================================
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🕘 Receipt History")
    st.caption("View, filter, and manage your receipts with categories and currency info")
    st.markdown("</div>", unsafe_allow_html=True)

    df = db_load_all()

    if len(df) == 0:
        st.info("No history found. Upload receipts in Tab 1.")
    else:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Search / Filter")

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            merchants = sorted(df["merchant"].fillna("").unique().tolist())
            merchant_filter = st.selectbox("Filter by Merchant", ["ALL"] + merchants, key="merchant_filter")
        
        with filter_col2:
            if "category" in df.columns:
                categories = sorted(df["category"].fillna("Others").unique().tolist())
                category_filter = st.selectbox("Filter by Category", ["ALL"] + categories, key="category_filter")
            else:
                category_filter = "ALL"
        
        with filter_col3:
            if "currency" in df.columns:
                currencies = sorted(df["currency"].fillna("INR").unique().tolist())
                currency_filter = st.selectbox("Filter by Currency", ["ALL"] + currencies, key="currency_filter")
            else:
                currency_filter = "ALL"

        valid_dates = df["parsed_date"].dropna()
        min_date = valid_dates.min().date() if len(valid_dates) > 0 else datetime.today().date()
        max_date = valid_dates.max().date() if len(valid_dates) > 0 else datetime.today().date()
        date_range = st.date_input("Filter by Date Range", value=(min_date, max_date))
        st.markdown("</div>", unsafe_allow_html=True)

        filtered_df = df.copy()
        
        if merchant_filter != "ALL":
            filtered_df = filtered_df[filtered_df["merchant"] == merchant_filter]
        
        if category_filter != "ALL":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]
        
        if currency_filter != "ALL":
            filtered_df = filtered_df[filtered_df["currency"] == currency_filter]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_df = filtered_df[
                (filtered_df["parsed_date"] >= start_dt) &
                (filtered_df["parsed_date"] <= end_dt)
            ]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🗑️ Delete Receipts")
        delete_ids = st.multiselect("Select Receipt ID(s) to delete", options=filtered_df["id"].tolist(), key="delete_multiselect")
        if st.button("❌ Delete Selected", key="delete_button"):
            if delete_ids:
                db_delete_by_ids(delete_ids)
                st.success(f"Deleted receipt IDs: {delete_ids} ✅")
                st.rerun()
            else:
                st.warning("Please select at least one receipt ID.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📌 Persistent Storage")
        
        display_df = filtered_df.copy()
        display_df["date"] = display_df["date"].apply(lambda x: str(x).split()[0] if pd.notna(x) and str(x).strip() else "")
        display_df["time"] = display_df["time"].fillna("")
        
        # Drop unnecessary columns - FIXED: Remove items_json, parsed_date
        columns_to_drop = ["items_json", "parsed_date"]
        columns_to_drop = [c for c in columns_to_drop if c in display_df.columns]
        
        st.dataframe(display_df.drop(columns=columns_to_drop), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        if len(filtered_df) > 0:
            selected = st.selectbox("Select receipt ID for details", filtered_df["id"].tolist(), key="history_select")
            rec = filtered_df[filtered_df["id"] == selected].iloc[0]

            # FIXED: Ensure items are loaded fresh each time
            try:
                items_json_str = rec["items_json"]
                if items_json_str and items_json_str != "[]":
                    items = pd.read_json(StringIO(items_json_str))
                else:
                    items = pd.DataFrame(columns=["name", "qty", "price"])
            except:
                items = pd.DataFrame(columns=["name", "qty", "price"])

            items_count = int(items["qty"].sum()) if "qty" in items and len(items) > 0 else 0

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧾 Detailed Bill")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Merchant", rec["merchant"] if rec["merchant"] else "N/A")
            c2.metric("Date", rec["date"] if rec["date"] else "N/A")
            c3.metric("Time", rec.get("time", "") if rec.get("time", "") else "N/A")
            c4.metric("Receipt ID", rec["id"])
            c5.metric("Payment", rec["payment"] if rec["payment"] else "N/A")
            
            # Show category badge
            if "category" in rec:
                category = rec["category"] if rec["category"] else "Others"
                cat_class = f"cat-{category.lower()}"
                c6.markdown(f'<div style="text-align: center;"><small>Category</small><br><span class="category-badge {cat_class}">{category}</span></div>', unsafe_allow_html=True)

            if st.button("🗑️ Delete This Receipt", key=f"delete_single_{rec['id']}"):
                db_delete_by_ids([int(rec["id"])])
                st.success(f"Deleted Receipt ID: {rec['id']} ✅")
                st.rerun()

            st.markdown("#### 🧺 Line Items")
            if len(items) > 0:
                st.table(items)
            else:
                st.info("No line items available")

            st.markdown("#### 💰 Summary")
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            
            currency = rec.get("currency", "INR")
            currency_symbol = CURRENCY_SYMBOLS.get(currency, "₹")
            
            s1.metric("Subtotal", f"{currency_symbol} {safe_float(rec['subtotal']):.2f}")
            s2.metric("Tax", f"{currency_symbol} {safe_float(rec['tax']):.2f}")
            
            tip_value = safe_float(rec.get('tip', 0))
            if tip_value > 0:
                s3.metric("Tip", f"{currency_symbol} {tip_value:.2f}")
            else:
                s3.metric("Tip", f"{currency_symbol} 0.00")
            
            s4.metric("Total", f"{currency_symbol} {safe_float(rec['total']):.2f}")
            s5.metric("Total (INR)", f"₹ {safe_float(rec.get('total_inr', rec['total'])):.2f}")
            s6.metric("Items", items_count)

            st.markdown("</div>", unsafe_allow_html=True)
# ==========================================
# TAB 3: ANALYTICS DASHBOARD - ENHANCED
# ==========================================
with tabs[2]:
    df = db_load_all()

    if len(df) == 0:
        st.info("No data available for analytics. Upload receipts first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Advanced Analytics Dashboard")
        st.caption("Multi-currency • Category insights • Interactive visualizations")
        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # DATA PREPARATION
        # ==========================================
        dfx = df.copy()
        dfx["total"] = pd.to_numeric(dfx["total"], errors="coerce").fillna(0)
        dfx["subtotal"] = pd.to_numeric(dfx["subtotal"], errors="coerce").fillna(0)
        dfx["tax"] = pd.to_numeric(dfx["tax"], errors="coerce").fillna(0)
        dfx["tip"] = pd.to_numeric(dfx.get("tip", 0), errors="coerce").fillna(0)
        dfx["total_inr"] = pd.to_numeric(dfx.get("total_inr", dfx["total"]), errors="coerce").fillna(0)
        dfx["timestamp"] = pd.to_datetime(dfx["created_at"], errors="coerce")
        dfx["date_only"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx["merchant"] = dfx["merchant"].fillna("Unknown").astype(str).str.strip()
        dfx["payment"] = dfx["payment"].fillna("CASH").astype(str).str.strip()
        dfx["category"] = dfx.get("category", "Others").fillna("Others").astype(str).str.strip()
        dfx["currency"] = dfx.get("currency", "INR").fillna("INR").astype(str).str.strip()
        
        dfx["year"] = dfx["date_only"].dt.year
        dfx["month"] = dfx["date_only"].dt.month
        dfx["month_name"] = dfx["date_only"].dt.strftime("%B")
        dfx["year_month"] = dfx["date_only"].dt.strftime("%Y-%m")
        dfx["quarter"] = dfx["date_only"].dt.quarter
        dfx["day_of_week"] = dfx["date_only"].dt.day_name()
        
        dfx = dfx[dfx["total"] > 0]

        # ==========================================
        # ENHANCED FILTERS
        # ==========================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analytics Filters")
        
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
        
        with col_f1:
            merchants_list = ["ALL"] + sorted(dfx["merchant"].unique().tolist())
            selected_merchant = st.selectbox("Merchant", merchants_list, key="analytics_merchant")
        
        with col_f2:
            payment_list = ["ALL"] + sorted(dfx["payment"].unique().tolist())
            selected_payment = st.selectbox("Payment", payment_list, key="analytics_payment")
        
        with col_f3:
            category_list = ["ALL"] + sorted(dfx["category"].unique().tolist())
            selected_category = st.selectbox("Category", category_list, key="analytics_category")
        
        with col_f4:
            currency_list = ["ALL"] + sorted(dfx["currency"].unique().tolist())
            selected_currency = st.selectbox("Currency", currency_list, key="analytics_currency")
        
        with col_f5:
            years_list = ["ALL"] + sorted(dfx["year"].dropna().unique().astype(int).astype(str).tolist(), reverse=True)
            selected_year = st.selectbox("Year", years_list, key="analytics_year")
        
        col_f6, col_f7 = st.columns(2)
        with col_f6:
            valid_dates = dfx["date_only"].dropna()
            min_date = valid_dates.min().date() if len(valid_dates) > 0 else datetime.today().date()
            max_date = valid_dates.max().date() if len(valid_dates) > 0 else datetime.today().date()
            date_range_analytics = st.date_input("Date Range", value=(min_date, max_date), key="analytics_date_range")
        
        with col_f7:
            min_total = float(dfx["total_inr"].min())
            max_total = float(dfx["total_inr"].max())

            if min_total == max_total:
                st.info(f"Only one total: ₹ {min_total:.2f}")
                total_range = (min_total, max_total)
            else:
                total_range = st.slider(
                    "Total Range (INR)",
                    min_total, max_total, (min_total, max_total),
                    key="analytics_total_range"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # APPLY FILTERS
        # ==========================================
        filtered_analytics_df = dfx.copy()
        
        if selected_merchant != "ALL":
            filtered_analytics_df = filtered_analytics_df[filtered_analytics_df["merchant"] == selected_merchant]
        
        if selected_payment != "ALL":
            filtered_analytics_df = filtered_analytics_df[filtered_analytics_df["payment"] == selected_payment]
        
        if selected_category != "ALL":
            filtered_analytics_df = filtered_analytics_df[filtered_analytics_df["category"] == selected_category]
        
        if selected_currency != "ALL":
            filtered_analytics_df = filtered_analytics_df[filtered_analytics_df["currency"] == selected_currency]
        
        if selected_year != "ALL":
            filtered_analytics_df = filtered_analytics_df[filtered_analytics_df["year"] == int(selected_year)]
        
        if isinstance(date_range_analytics, tuple) and len(date_range_analytics) == 2:
            start_date, end_date = date_range_analytics
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_analytics_df = filtered_analytics_df[
                (filtered_analytics_df["date_only"] >= start_dt) &
                (filtered_analytics_df["date_only"] <= end_dt)
            ]
        
        filtered_analytics_df = filtered_analytics_df[
            (filtered_analytics_df["total_inr"] >= total_range[0]) &
            (filtered_analytics_df["total_inr"] <= total_range[1])
        ]

        # ==========================================
        # KEY METRICS
        # ==========================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Key Metrics (All amounts in INR)")
        st.caption("💡 **Quick Summary**: Overall spending patterns normalized to INR for comparison")
        
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
        
        with metric_col1:
            total_spent = filtered_analytics_df["total_inr"].sum()
            st.metric("Total Spent", f"₹ {total_spent:,.2f}")
            st.caption("💰 Total in INR")
        
        with metric_col2:
            avg_transaction = filtered_analytics_df["total_inr"].mean()
            st.metric("Avg Transaction", f"₹ {avg_transaction:,.2f}")
            st.caption("📊 Average per receipt")
        
        with metric_col3:
            total_receipts = len(filtered_analytics_df)
            st.metric("Total Receipts", f"{total_receipts}")
            st.caption("🧾 Number of purchases")
        
        with metric_col4:
            # Calculate tax in INR
            tax_inr = (filtered_analytics_df["tax"] * filtered_analytics_df["currency"].map(EXCHANGE_RATES)).sum()
            st.metric("Total Tax", f"₹ {tax_inr:,.2f}")
            st.caption("🏛️ Tax contributed")
        
        with metric_col5:
            unique_merchants = filtered_analytics_df["merchant"].nunique()
            st.metric("Merchants", f"{unique_merchants}")
            st.caption("🏪 Different stores")
        
        with metric_col6:
            unique_categories = filtered_analytics_df["category"].nunique()
            st.metric("Categories", f"{unique_categories}")
            st.caption("🏷️ Spending types")
        
        st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # VISUALIZATIONS
        # ==========================================
        if len(filtered_analytics_df) > 0:
            
            # ========== CATEGORY ANALYSIS ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏷️ Category Analysis")
            st.info("📌 **What this tells you**: See how your spending is distributed across different categories")
            
            cat_col1, cat_col2 = st.columns(2)
            
            with cat_col1:
                st.markdown("#### Spending by Category (Treemap)")
                category_spending = filtered_analytics_df.groupby("category")["total_inr"].sum().reset_index()
                category_spending = category_spending.sort_values("total_inr", ascending=False)
                
                if len(category_spending) > 0:
                    fig_treemap = px.treemap(
                        category_spending,
                        path=['category'],
                        values='total_inr',
                        title='',
                        color='total_inr',
                        color_continuous_scale='Viridis'
                    )
                    fig_treemap.update_layout(height=400)
                    st.plotly_chart(fig_treemap, width="stretch")
                    
                    top_category = category_spending.iloc[0]
                    st.success(f"💡 **Insight**: **{top_category['category']}** is your top spending category (₹{top_category['total_inr']:,.2f})")
            
            with cat_col2:
                st.markdown("#### Category Distribution")
                if len(category_spending) > 0:
                    fig_pie = px.pie(
                        category_spending,
                        values='total_inr',
                        names='category',
                        title='',
                        hole=0.4
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, width="stretch")
                    
                    percentages = (category_spending["total_inr"] / category_spending["total_inr"].sum() * 100).round(1)
                    top_3_cats = ", ".join([f"{cat} ({pct}%)" for cat, pct in zip(category_spending["category"][:3], percentages[:3])])
                    st.success(f"💡 **Insight**: Top 3 categories: {top_3_cats}")
            
            st.markdown("</div>", unsafe_allow_html=True)
# ========== CATEGORY TRENDS ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Category Trends Over Time")
            st.info("📌 **What this tells you**: Track how spending in each category changes month by month")
            
            category_monthly = filtered_analytics_df.groupby(["year_month", "category"])["total_inr"].sum().reset_index()
            
            if len(category_monthly) > 0:
                fig_trends = px.line(
                    category_monthly,
                    x="year_month",
                    y="total_inr",
                    color="category",
                    title="Monthly Spending by Category",
                    markers=True
                )
                fig_trends.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Amount (₹)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trends, width="stretch")
                
                # Find trending category
                latest_month = category_monthly["year_month"].max()
                latest_data = category_monthly[category_monthly["year_month"] == latest_month]
                if len(latest_data) > 0:
                    trending_cat = latest_data.sort_values("total_inr", ascending=False).iloc[0]
                    st.success(f"💡 **Insight**: In {latest_month}, **{trending_cat['category']}** had highest spending (₹{trending_cat['total_inr']:,.2f})")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # ========== PAYMENT vs CATEGORY HEATMAP ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔥 Payment Method vs Category Heatmap")
            st.info("📌 **What this tells you**: See which payment methods you use for different categories")
            
            payment_category = filtered_analytics_df.groupby(["payment", "category"])["total_inr"].sum().reset_index()
            payment_category_pivot = payment_category.pivot(index="category", columns="payment", values="total_inr").fillna(0)
            
            if not payment_category_pivot.empty:
                fig_heatmap = px.imshow(
                    payment_category_pivot,
                    labels=dict(x="Payment Method", y="Category", color="Amount (₹)"),
                    title="",
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, width="stretch")
                
                # Find dominant payment-category combo
                max_combo = payment_category.sort_values("total_inr", ascending=False).iloc[0]
                st.success(f"💡 **Insight**: You mostly use **{max_combo['payment']}** for **{max_combo['category']}** (₹{max_combo['total_inr']:,.2f})")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # ========== CURRENCY DISTRIBUTION ==========
            if len(filtered_analytics_df["currency"].unique()) > 1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 💱 Multi-Currency Analysis")
                st.info("📌 **What this tells you**: See which currencies you use and their INR equivalent")
                
                curr_col1, curr_col2 = st.columns(2)
                
                with curr_col1:
                    st.markdown("#### Spending by Currency")
                    currency_dist = filtered_analytics_df.groupby("currency").agg({
                        "total": "sum",
                        "total_inr": "sum"
                    }).reset_index()
                    
                    fig_curr = px.bar(
                        currency_dist,
                        x="currency",
                        y="total_inr",
                        title="",
                        text="total_inr",
                        color="currency"
                    )
                    fig_curr.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                    fig_curr.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_curr, width="stretch")
                
                with curr_col2:
                    st.markdown("#### Currency Breakdown")
                    for _, row in currency_dist.iterrows():
                        curr_symbol = CURRENCY_SYMBOLS.get(row["currency"], row["currency"])
                        st.metric(
                            f"{row['currency']} Spending",
                            f"{curr_symbol} {row['total']:,.2f}",
                            delta=f"₹ {row['total_inr']:,.2f} INR"
                        )
                
                st.markdown("</div>", unsafe_allow_html=True)

            # ========== MERCHANT ANALYSIS ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏪 Top Merchants")
            st.info("📌 **What this tells you**: Your most visited and highest spending merchants")
            
            merch_col1, merch_col2 = st.columns(2)
            
            with merch_col1:
                st.markdown("#### Top 10 by Spending")
                merchant_spending = filtered_analytics_df.groupby("merchant")["total_inr"].sum().sort_values(ascending=False).head(10)
                
                fig_merch = px.bar(
                    x=merchant_spending.values,
                    y=merchant_spending.index,
                    orientation='h',
                    title="",
                    labels={"x": "Total (₹)", "y": "Merchant"}
                )
                fig_merch.update_traces(marker_color='#60a5fa')
                fig_merch.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_merch, width="stretch")
                
                if len(merchant_spending) > 0:
                    top_merch = merchant_spending.index[0]
                    top_amt = merchant_spending.values[0]
                    st.success(f"💡 **Insight**: **{top_merch}** is your top expense (₹{top_amt:,.2f})")
            
            with merch_col2:
                st.markdown("#### Top 10 by Visits")
                merchant_count = filtered_analytics_df.groupby("merchant").size().sort_values(ascending=False).head(10)
                
                fig_visits = px.bar(
                    x=merchant_count.index,
                    y=merchant_count.values,
                    title="",
                    labels={"x": "Merchant", "y": "Visit Count"}
                )
                fig_visits.update_traces(marker_color='#a78bfa')
                fig_visits.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_visits, width="stretch")
                
                if len(merchant_count) > 0:
                    frequent = merchant_count.index[0]
                    visits = merchant_count.values[0]
                    st.success(f"💡 **Insight**: You visit **{frequent}** most ({visits} times)")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # ========== TIME ANALYSIS ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📅 Time-Based Patterns")
            st.info("📌 **What this tells you**: When you spend the most - monthly and weekly patterns")
            
            time_col1, time_col2 = st.columns(2)
            
            with time_col1:
                st.markdown("#### Monthly Trend")
                monthly_spending = filtered_analytics_df.groupby("year_month")["total_inr"].sum().sort_index()
                
                if len(monthly_spending) > 0:
                    fig_monthly = px.area(
                        x=monthly_spending.index,
                        y=monthly_spending.values,
                        title="",
                        labels={"x": "Month", "y": "Amount (₹)"}
                    )
                    fig_monthly.update_traces(line_color='#34d399', fillcolor='rgba(52, 211, 153, 0.3)')
                    fig_monthly.update_layout(height=350)
                    st.plotly_chart(fig_monthly, width="stretch")
                    
                    max_month = monthly_spending.idxmax()
                    max_amt = monthly_spending.max()
                    st.success(f"💡 **Insight**: Peak spending in **{max_month}** (₹{max_amt:,.2f})")
            
            with time_col2:
                st.markdown("#### Day of Week Pattern")
                dow_spending = filtered_analytics_df.groupby("day_of_week")["total_inr"].sum()
                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow_spending = dow_spending.reindex([d for d in dow_order if d in dow_spending.index])
                
                if len(dow_spending) > 0:
                    fig_dow = px.bar(
                        x=dow_spending.index,
                        y=dow_spending.values,
                        title="",
                        labels={"x": "Day", "y": "Amount (₹)"}
                    )
                    fig_dow.update_traces(marker_color=['#60a5fa', '#a78bfa', '#34d399', '#fbbf24', '#f87171', '#fb923c', '#ec4899'][:len(dow_spending)])
                    fig_dow.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig_dow, width="stretch")
                    
                    max_day = dow_spending.idxmax()
                    st.success(f"💡 **Insight**: You spend most on **{max_day}**")
            
            st.markdown("</div>", unsafe_allow_html=True)
# ========== DATA TABLE ==========
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Filtered Data Summary")
            st.caption("💡 **What this shows**: All receipts matching your filters above")
            
            summary_df = filtered_analytics_df[["id", "merchant", "category", "date", "time", "payment", "currency", "subtotal", "tax", "tip", "total", "total_inr"]].copy()
            summary_df["date"] = summary_df["date"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "")
            
            st.dataframe(summary_df, width="stretch", height=300)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("No data available after applying filters.")

        # ==========================================
        # ENHANCED EXPORT SECTION
        # ==========================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📥 Export Data")
        st.caption("💡 **Why export?**: Share with accountants, track in Excel, or keep for tax records")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        export_df = filtered_analytics_df[["id", "created_at", "merchant", "category", "date", "time", 
                                           "payment", "currency", "subtotal", "tax", "tip", "total", "total_inr"]].copy()
        export_df["date"] = export_df["date"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "")
        
        with export_col1:
            st.markdown("#### 📄 CSV Export")
            st.caption("✅ Best for: Excel, Google Sheets")
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"receipts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with export_col2:
            st.markdown("#### 📊 Excel Export")
            st.caption("✅ Best for: Professional reports")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Receipts')
                
                # Summary sheet
                summary_stats = pd.DataFrame({
                    'Metric': ['Total Receipts', 'Total Spent (INR)', 'Avg Transaction', 
                              'Total Tax', 'Unique Merchants', 'Categories Used'],
                    'Value': [
                        len(filtered_analytics_df),
                        f"₹ {filtered_analytics_df['total_inr'].sum():,.2f}",
                        f"₹ {filtered_analytics_df['total_inr'].mean():,.2f}",
                        f"₹ {(filtered_analytics_df['tax'] * filtered_analytics_df['currency'].map(EXCHANGE_RATES)).sum():,.2f}",
                        filtered_analytics_df['merchant'].nunique(),
                        filtered_analytics_df['category'].nunique()
                    ]
                })
                summary_stats.to_excel(writer, index=False, sheet_name='Summary')
                
                # Category breakdown
                cat_breakdown = filtered_analytics_df.groupby('category')['total_inr'].sum().reset_index()
                cat_breakdown.columns = ['Category', 'Total (INR)']
                cat_breakdown = cat_breakdown.sort_values('Total (INR)', ascending=False)
                cat_breakdown.to_excel(writer, index=False, sheet_name='Category Breakdown')
            
            excel_buffer.seek(0)
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_buffer,
                file_name=f"receipts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )
        
        with export_col3:
            st.markdown("#### 📑 Custom Report")
            st.caption("✅ Best for: Quick overview")
            
            report_text = f"""
RECEIPT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SUMMARY STATISTICS (All amounts in INR):
- Total Receipts: {len(filtered_analytics_df)}
- Total Spent: ₹ {filtered_analytics_df['total_inr'].sum():,.2f}
- Average Transaction: ₹ {filtered_analytics_df['total_inr'].mean():,.2f}
- Total Tax: ₹ {(filtered_analytics_df['tax'] * filtered_analytics_df['currency'].map(EXCHANGE_RATES)).sum():,.2f}
- Unique Merchants: {filtered_analytics_df['merchant'].nunique()}
- Categories: {filtered_analytics_df['category'].nunique()}

TOP 5 MERCHANTS BY SPENDING:
"""
            top_merchants = filtered_analytics_df.groupby("merchant")["total_inr"].sum().sort_values(ascending=False).head(5)
            for idx, (merchant, amount) in enumerate(top_merchants.items(), 1):
                report_text += f"{idx}. {merchant}: ₹ {amount:,.2f}\n"
            
            report_text += f"""
SPENDING BY CATEGORY:
"""
            cat_spending = filtered_analytics_df.groupby("category")["total_inr"].sum().sort_values(ascending=False)
            for category, amount in cat_spending.items():
                percentage = (amount / filtered_analytics_df['total_inr'].sum()) * 100
                report_text += f"- {category}: ₹ {amount:,.2f} ({percentage:.1f}%)\n"
            
            report_text += f"""
PAYMENT METHOD BREAKDOWN:
"""
            payment_breakdown = filtered_analytics_df.groupby("payment")["total_inr"].sum()
            for payment, amount in payment_breakdown.items():
                percentage = (amount / filtered_analytics_df['total_inr'].sum()) * 100
                report_text += f"- {payment}: ₹ {amount:,.2f} ({percentage:.1f}%)\n"
            
            if len(filtered_analytics_df["currency"].unique()) > 1:
                report_text += f"""
CURRENCY BREAKDOWN:
"""
                curr_breakdown = filtered_analytics_df.groupby("currency").agg({
                    "total": "sum",
                    "total_inr": "sum"
                })
                for curr, row in curr_breakdown.iterrows():
                    curr_symbol = CURRENCY_SYMBOLS.get(curr, curr)
                    report_text += f"- {curr}: {curr_symbol} {row['total']:,.2f} (₹ {row['total_inr']:,.2f} INR)\n"
            
            report_text += f"""
{'='*60}
End of Report
"""
            
            st.download_button(
                label="⬇️ Download Report (TXT)",
                data=report_text,
                file_name=f"receipt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width="stretch"
            )
        
        st.info("📌 **Tip**: All exports reflect your current filter selections. Change filters to export different data!")
        st.markdown("</div>", unsafe_allow_html=True)
