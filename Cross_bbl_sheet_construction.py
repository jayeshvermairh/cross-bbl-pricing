# libraries that we require
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd
import json
import re
import smtplib
from email.message import EmailMessage
import os

# Month code to name mapping used in futures contracts (standardisation)
month_map = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"
}

# invert the dictionary - so that it identifies the correct month
month_name_to_code = {v: k for k, v in month_map.items()}

## Generate futures contract symbols for a given base code from start_date forward.
def parse_barchart_symbols(base_code: str, start_year=None, end_year=None, start_month=None):
    # Parameters:
    # - base_code (str): Ticker root (e.g., "RB" for RBOB, "IIH" for 0.5% Sing Fuel Oil).
    # - start_year (int, optional): Start year (4-digit). Defaults to current year.
    # - end_year (int, optional): End year (4-digit). Defaults to 2 years ahead.
    # - start_month (str, optional): Starting month code (e.g., "F" for Jan). Defaults to current month.

    # Returns:
    # - List[str]: List of symbols like ["RBQ25", "RBU25", ..., "RBZ27"]
    # We plug this list into the subsequent function to scrape price data from barchart.com wrt the contract symbol

    today = datetime.today()

    # Set start year and month dynamically
    if start_year is None:
        start_year = today.year
    
    # Set default start month using current month abbreviation
    if start_month is None:
        current_month_abbr = today.strftime("%b")  # e.g., "Aug"
        start_month = month_name_to_code[current_month_abbr]

    # Calculate end year based on 24 months ahead
    if end_year is None:
        end_date = today + relativedelta(months=+24)
        end_year = end_date.year

    # Month codes used in futures tickers (Jan-Dec)
    month_codes = list(month_map.keys())

    # Build list of two-digit years between start and end
    years = list(range(start_year % 100, end_year % 100 + 1))

    symbols = []
    for y in years:
        for m in month_codes:
            # Skip earlier months in the start year
            if y == start_year % 100 and month_codes.index(m) < month_codes.index(start_month):
                continue
            symbols.append(f"{base_code}{m}{y}")
    return symbols

## Fetching prices form Barchart Symbol pages
def fetch_barchart_prices(symbols, base_url="https://www.barchart.com/futures/quotes/{}/overview"):

    #Scrape last traded futures prices from Barchart for a list of contract symbols.

    #Parameters:
    #- symbols (List[str]): List of Barchart futures symbols (e.g., ["RBV25", "RBX25"])
    #- base_url (str): URL template with `{}` where symbol gets inserted

    #Returns:
    #- pd.DataFrame: Contains columns ['symbol', 'last_price']
    
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    
    for symbol in symbols:
        url = base_url.format(symbol)
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            div = soup.find("div", class_="symbol-header-info")
            if not div:
                continue
            ng_init = div.get("data-ng-init")
            match = re.search(r'init\((\{.*?\})\)', ng_init)
            if not match:
                continue
            data = json.loads(match.group(1).replace(r'\/', '/'))

            raw_price = data.get("lastPrice")
            if raw_price in [None, "N/A", "-"]:
                continue
            clean_price = re.sub(r"[^\d.]+$", "", str(raw_price))

            records.append({
                "symbol": symbol,  # FIX: use passed symbol not unreliable data.get()
                "last_price": float(clean_price) if clean_price else None
            })
        except Exception:
            continue
    return pd.DataFrame(records)

## Abit of housekeeping/cleaning at the end to tie it all together
def clean_and_format_df(df_raw, label):
    df = df_raw.copy()

    # Determine base code length: 2 for RB, 3 for others
    # Assume all rows in df_raw have the same base length
    sample_symbol = df["symbol"].iloc[0]
    base_code_len = 2 if sample_symbol[:2] == "RB" else 3

    # Only keep symbols that are long enough to contain month code and year
    df = df[df["symbol"].str.len() >= base_code_len + 3]

    # Extract month code dynamically
    df["month_code"] = df["symbol"].str[base_code_len]
    df["month"] = df["month_code"].map(month_map)

    # Drop rows with unrecognized month codes (e.g., bad cash contracts)
    df = df[df["month"].notna()]

    # Extract year from last two characters
    df["year"] = 2000 + df["symbol"].str[-2:].astype(int)

    # Format output
    df = df[["month", "year", "last_price"]].rename(columns={"last_price": label})
    df["order"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"], format="%Y-%b")
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)

## ======================== Pulling the fuel curves (no 180-Sing sadly)========================

# 0.5 Sing flat price curve (IIH)
symbols_05 = parse_barchart_symbols("IIH")
df_05_raw = fetch_barchart_prices(symbols_05)
df_05 = clean_and_format_df(df_05_raw, "0.5-Sing flat ($/kt)")

# 380cst sing flat price curve (JSE)
symbols_380 = parse_barchart_symbols("JSE")
df_380_raw = fetch_barchart_prices(symbols_380)
df_380 = clean_and_format_df(df_380_raw, "380 Sing flat ($/kt)")

# 0.5% Euro Barges (IID)
symbols_iid = parse_barchart_symbols("IID")
df_iid_raw = fetch_barchart_prices(symbols_iid)
df_05_barge = clean_and_format_df(df_iid_raw, "0.5-Barges flat ($/kt)")

# 3.5% ARA Barges (JUV)
symbols_juv = parse_barchart_symbols("JUV")
df_juv_raw = fetch_barchart_prices(symbols_juv)
df_35_barge = clean_and_format_df(df_juv_raw, "3.5-Barges flat ($/kt)")

## List all your DataFrames
dfs = [df_05, df_380, df_05_barge, df_35_barge]

# Merge them one by one on 'month' and 'year'
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on=["month", "year"], how="outer")

# Sort for neatness
merged_df = merged_df.sort_values(by=["year", "month"]).reset_index(drop=True)
merged_df

# Create a datetime column from month and year
merged_df["date"] = pd.to_datetime(merged_df["month"] + " " + merged_df["year"].astype(str), format="%b %Y")

# Sort by the datetime column
merged_df = merged_df.sort_values("date").drop(columns="date").reset_index(drop=True)


## ======================== Deriving Implied brent swap curve (using 3.5 Euro Barge flat and crack) ========================

# pulling 3.5 bdge crk (can be recycled later for our curves)
symbols_jfo = parse_barchart_symbols("JFO")
df_jfo_raw = fetch_barchart_prices(symbols_jfo)
df_35_bdg_crack = clean_and_format_df(df_jfo_raw, "3.5 Bdge Crk ($/bbl)")
#df_35_bdg_crack

merged_df = pd.merge(merged_df, df_35_bdg_crack, on=["month", "year"], how="left")

# Convert 3.5% Barge flat price from $/kt to $/bbl
merged_df["3.5 bdge flat ($/bbl)"] = merged_df["3.5-Barges flat ($/kt)"] / 6.35

# Calculate Brent swap by subtracting crack from flat price in $/bbl 
merged_df["Implied Brent swap ($/bbl)"] = merged_df["3.5 bdge flat ($/bbl)"] - merged_df["3.5 Bdge Crk ($/bbl)"]


## ======================== Pulling the mid-disti curves (no Sing Kero) ========================

# Sing 10ppm pull
symbols_jsg = parse_barchart_symbols("JSG")
df_jsg_raw = fetch_barchart_prices(symbols_jsg)
df_10ppm = clean_and_format_df(df_jsg_raw, "10ppm flat ($/bbl)")

# LSGO flat price
symbols_lf = parse_barchart_symbols("LF")
df_lf_raw = fetch_barchart_prices(symbols_lf)

# Special Housekeeping for the LSGO curve
def clean_lf_format(df_raw, label):
    df = df_raw.copy()
    df["month_code"] = df["symbol"].str[2]  # Month at index 2 for 'LFU25'
    df["month"] = df["month_code"].map(month_map)
    df["year"] = 2000 + df["symbol"].str[-2:].astype(int)
    df = df[["month", "year", "last_price"]].rename(columns={"last_price": label})
    df["order"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"], format="%Y-%b", errors='coerce')
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)

df_lsgo_flat = clean_lf_format(df_lf_raw, "LSGO flat ($/kt)")

# Adding these curves to main dataframe
for df in [df_10ppm, df_lsgo_flat]:
    merged_df = pd.merge(merged_df, df, on=["month", "year"], how="left")

## ======================== Pulling the Gasoline curves ========================

# M92 scrape
symbols_m92 = parse_barchart_symbols("J1N")
df_m92_raw = fetch_barchart_prices(symbols_m92)
df_m92 = clean_and_format_df(df_m92_raw, "M92 flat ($/bbl)")

# EBOB Scrape 
symbols_ebob = parse_barchart_symbols("J7H")
df_ebob_raw = fetch_barchart_prices(symbols_ebob)
df_ebob = clean_and_format_df(df_ebob_raw, "EBOB flat ($/kt)")

# RBOB Scrape
# Get the month after the current month (this month has already settled) 
# - RBOB future is similar to brent it skips a month 
next_month_abbr = (datetime.today() + relativedelta(months=+1)).strftime("%b")
next_month_code = month_name_to_code[next_month_abbr]

symbols_rbob = parse_barchart_symbols("RB", start_month=next_month_code)
df_rbob_raw = fetch_barchart_prices(symbols_rbob)
df_rbob = clean_and_format_df(df_rbob_raw, "RBOB flat ($/gal)")

# Merge gasoline-related flat price curves
for df in [df_m92, df_ebob, df_rbob]:
    merged_df = pd.merge(merged_df, df, on=["month", "year"], how="left")

## ======================== Pulling the MOPJ curve ========================

# Naphtha Scrpae
symbols_MOPJ = parse_barchart_symbols("JJA")
df_MOPJ_raw = fetch_barchart_prices(symbols_MOPJ)
df_MOPJ = clean_and_format_df(df_MOPJ_raw, "MOPJ flat ($/kt)")

merged_df = merged_df.merge(df_MOPJ, on=["month", "year"], how="left")

# Ensure sorting
merged_df["order"] = pd.to_datetime(merged_df["year"].astype(int).astype(str) + "-" + merged_df["month"], format="%Y-%b")
merged_df = merged_df.sort_values("order").reset_index(drop=True)

## ======================== Computing M1/M2 Timespreads ========================

# Compute Jul–Aug style time spreads
col = "0.5-Sing flat ($/kt)"
ts_col = "0.5-Sing TS ($/kt)"

merged_df[ts_col] = merged_df.apply(
    lambda x: x[col] - merged_df[col].shift(-1).loc[x.name]
    if pd.notnull(x[col]) and pd.notnull(merged_df[col].shift(-1).loc[x.name])
    else np.nan,
    axis=1
)

# Drop helper column
merged_df = merged_df.drop(columns="order")

# Ensure merged_df is sorted by date
merged_df["order"] = pd.to_datetime(
    merged_df["year"].astype(int).astype(str) + "-" + merged_df["month"],
    format="%Y-%b"
)
merged_df = merged_df.sort_values("order").reset_index(drop=True)

# List of flat price columns to calculate time spreads for
flat_columns = [
    "0.5-Sing flat ($/kt)", # remove code above for the 0.5-Sing flat tmrw
    "380 Sing flat ($/kt)",
    "0.5-Barges flat ($/kt)",
    "3.5-Barges flat ($/kt)",
    "10ppm flat ($/bbl)",
    "LSGO flat ($/kt)",
    "M92 flat ($/bbl)",
    "EBOB flat ($/kt)",
    "RBOB flat ($/gal)",
    "MOPJ flat ($/kt)"
]

# Compute TS for each and append new column
for col in flat_columns:
    ts_col = col.replace("flat", "TS")
    merged_df[ts_col] = merged_df[col] - merged_df[col].shift(-1)

# Drop sorting helper
merged_df = merged_df.drop(columns="order")

## ======================== Deriving Cracks ========================

# keeping a df prior to including cracks (in case of any fuck ups)
merged_df_for_crks = merged_df.copy()

# Conversion factors
conversion_factors = {
    "380 Sing flat ($/kt)": 6.35,
    "0.5-Barges flat ($/kt)": 6.35,
    "0.5-Sing flat ($/kt)": 6.35,
    "LSGO flat ($/kt)": 7.45,
    "EBOB flat ($/kt)": 8.33,
    "MOPJ flat ($/kt)": 8.9
}

# No conversion needed
no_conversion = ["10ppm flat ($/bbl)", "M92 flat ($/bbl)"]

## Maybe create a separate list or dictionary for RBOB as it prices in $/gal 
gal_conversion_factors = {
    "RBOB flat ($/gal)": 42.0 # implement for cracks then have to do the kt conversion for the "Arb"
    ##  Note:
    # RBBRs are priced in $/bbl, 
    # the "ARB (RBOB v EBOB)" priced in $/gal also give the $/kt & $/bbl cols as well
    # E/W Gasoline priced in $/bbl
}

# Special scenario for the 
for col in gal_conversion_factors:
    if col in merged_df_for_crks.columns:
        crack_col = col.replace("flat", "crk").replace("$/gal", "$/bbl").strip()
        merged_df_for_crks[crack_col] = merged_df_for_crks.apply(
            lambda row: (row[col] * gal_conversion_factors[col]) - row["Implied Brent swap ($/bbl)"]
            if pd.notnull(row[col]) and pd.notnull(row["Implied Brent swap ($/bbl)"]) else np.nan,
            axis=1
        )

# Calculate crack spreads
for col in conversion_factors:
    if col in merged_df_for_crks.columns:
        crack_col = col.replace("flat", "crk").replace("(kt)", "($/bbl)").strip()
        merged_df_for_crks[crack_col] = merged_df_for_crks.apply(
            lambda row: (row[col] / conversion_factors[col]) - row["Implied Brent swap ($/bbl)"]
            if pd.notnull(row[col]) and pd.notnull(row["Implied Brent swap ($/bbl)"]) else np.nan,
            axis=1
        )

for col in no_conversion:
    if col in merged_df_for_crks.columns:
        crack_col = col.replace("flat", "crk").strip()
        merged_df_for_crks[crack_col] = merged_df_for_crks.apply(
            lambda row: row[col] - row["Implied Brent swap ($/bbl)"]
            if pd.notnull(row[col]) and pd.notnull(row["Implied Brent swap ($/bbl)"]) else np.nan,
            axis=1
            )

# Final output
merged_df_for_blends = merged_df_for_crks.copy() # keep merged_df_for_crks as a reference in case of any fuck ups

## ======================== Cross Barrel Blends ========================

# Add each of the 92-MOPJ curves
merged_df_for_blends["M92 v MOPJ ($/bbl) 8.9-conv"] = merged_df_for_blends.apply(
    lambda row: row["M92 flat ($/bbl)"] - (row["MOPJ flat ($/kt)"]/8.9)
    if pd.notnull(row["M92 flat ($/bbl)"]) and pd.notnull(row["MOPJ flat ($/kt)"]) else np.nan,
    axis=1
)

merged_df_for_blends["M92 v MOPJ ($/bbl) 9.0-conv"] = merged_df_for_blends.apply(
    lambda row: row["M92 flat ($/bbl)"] - (row["MOPJ flat ($/kt)"]/9.0)
    if pd.notnull(row["M92 flat ($/bbl)"]) and pd.notnull(row["MOPJ flat ($/kt)"]) else np.nan,
    axis=1
)

merged_df_for_geo = merged_df_for_blends.copy() # keep copy df in case of fuckups
# merged_df_geo to add the E/W and geo arbs

## ======================== Geographical Spreads ========================
merged_df_for_geo["0.5 E/W ($/kt)"] = merged_df_for_geo.apply(
    lambda row: row["0.5-Sing flat ($/kt)"] - row["0.5-Barges flat ($/kt)"]
    if pd.notnull(row["0.5-Sing flat ($/kt)"]) and pd.notnull(row["0.5-Barges flat ($/kt)"]) else np.nan,
    axis=1
)

merged_df_for_geo["380 E/W ($/kt)"] = merged_df_for_geo.apply(
    lambda row: row["380 Sing flat ($/kt)"] - row["3.5-Barges flat ($/kt)"]
    if pd.notnull(row["380 Sing flat ($/kt)"]) and pd.notnull(row["3.5-Barges flat ($/kt)"]) else np.nan,
    axis=1
)

merged_df_for_geo["Sing Hi-5 ($/kt)"] = merged_df_for_geo.apply(
    lambda row: row["0.5-Sing flat ($/kt)"] - row["380 Sing flat ($/kt)"]
    if pd.notnull(row["0.5-Sing flat ($/kt)"]) and pd.notnull(row["380 Sing flat ($/kt)"]) else np.nan,
    axis=1
)

## Additions we need to make for the Gasoline diffs 

# E/W Gasoline
merged_df_for_geo["E/W Gasoline ($/bbl)"] = merged_df_for_geo.apply(
   lambda row: row["M92 flat ($/bbl)"] - (row["EBOB flat ($/kt)"]/8.33)
   if pd.notnull(row["M92 flat ($/bbl)"]) and pd.notnull(row["EBOB flat ($/kt)"]) else np.nan,
   axis=1
)

# Arb on Gasoline (RBOB v EBOB) in $/gal
merged_df_for_geo["Gasoline ARB ~ rbob v ebob ($/gal)"] = merged_df_for_geo.apply(
   lambda row: row["RBOB flat ($/gal)"] - ((row["EBOB flat ($/kt)"]/8.33)/42)
   if pd.notnull(row["RBOB flat ($/gal)"]) and pd.notnull(row["EBOB flat ($/kt)"]) else np.nan,
   axis=1
)

# Arb on Gasoline (RBOB v EBOB) in $/bbl
merged_df_for_geo["Gasoline ARB ~ rbob v ebob ($/bbl)"] = merged_df_for_geo.apply(
   lambda row: (row["RBOB flat ($/gal)"]*42) - ((row["EBOB flat ($/kt)"]/8.33))
   if pd.notnull(row["RBOB flat ($/gal)"]) and pd.notnull(row["EBOB flat ($/kt)"]) else np.nan,
   axis=1
)

## ======================== Cleaning up ========================

# List of columns to keep (your finalized structure)
columns_to_keep = [
    'month', 'year',
    'Implied Brent swap ($/bbl)',
    # Fuel Group
    '0.5-Sing flat ($/kt)','0.5-Sing TS ($/kt)','0.5-Sing crk ($/kt)','Sing Hi-5 ($/kt)','380 Sing flat ($/kt)','380 Sing TS ($/kt)','380 Sing crk ($/kt)',
    '0.5 E/W ($/kt)','380 E/W ($/kt)','0.5-Barges flat ($/kt)','0.5-Barges TS ($/kt)','0.5-Barges crk ($/kt)','3.5-Barges flat ($/kt)','3.5-Barges TS ($/kt)',
    '3.5 Bdge Crk ($/bbl)',

    # Mid-disti Group
    '10ppm flat ($/bbl)', '10ppm TS ($/bbl)', '10ppm crk ($/bbl)', 'LSGO flat ($/kt)', 'LSGO TS ($/kt)', 'LSGO crk ($/kt)',

    # Gasoline Group
    'M92 flat ($/bbl)', 'M92 TS ($/bbl)', 'M92 crk ($/bbl)', 'E/W Gasoline ($/bbl)',
    'EBOB flat ($/kt)', 'EBOB TS ($/kt)', 'EBOB crk ($/kt)',
    'Gasoline ARB ~ rbob v ebob ($/gal)', 'Gasoline ARB ~ rbob v ebob ($/bbl)', 'RBOB flat ($/gal)', 'RBOB TS ($/gal)', 'RBOB crk ($/bbl)',

    # Naphtha Group (incl gasoline blends)
    "M92 v MOPJ ($/bbl) 8.9-conv", # this is the one usually referred to in the mkt 
    "M92 v MOPJ ($/bbl) 9.0-conv",
    "MOPJ flat ($/kt)", "MOPJ TS ($/kt)", "MOPJ crk ($/kt)"
]

# Drop everything else
master_df = merged_df_for_geo[columns_to_keep]

# Correct mislabeled crack columns from ($/kt) to ($/bbl)
master_df.rename(columns={
    '0.5-Sing crk ($/kt)': '0.5-Sing crk ($/bbl)',
    '380 Sing crk ($/kt)': '380 Sing crk ($/bbl)',
    '0.5-Barges crk ($/kt)': '0.5-Barges crk ($/bbl)',
    'LSGO crk ($/kt)': 'LSGO crk ($/bbl)',
    'EBOB crk ($/kt)': 'EBOB crk ($/bbl)',
    'MOPJ crk ($/kt)': 'MOPJ crk ($/bbl)',
    'RBOB crk ($/bbl)': 'RBOB crk (RBBR) ~ ($/bbl)'
}, inplace=True)

# Define special columns for 4 decimal places
rbob_cols_4dp = [
    'RBOB flat ($/gal)',
    'RBOB TS ($/gal)',
    'Gasoline ARB ~ rbob v ebob ($/gal)'
]

# Get all other numeric columns (excluding 'month', 'year', and rbob_cols_4dp)
cols_to_format_2dp = master_df.columns.difference(['month', 'year'] + rbob_cols_4dp)

# Apply 2 decimal place rounding
master_df[cols_to_format_2dp] = master_df[cols_to_format_2dp].applymap(
    lambda x: round(x, 2) if pd.notnull(x) else x
)

# Apply 4 decimal place rounding to RBOB-related columns
master_df[rbob_cols_4dp] = master_df[rbob_cols_4dp].applymap(
    lambda x: round(x, 4) if pd.notnull(x) else x
) 

## ======================== Standardising Tenors - better readability for the sheet ========================

# creating the tenor column - better readability
master_df["tenor"] = master_df.apply(
    lambda row: f"{row['month'][:3]}'{str(row['year'])[-2:]}", axis=1
)

# drop the original columns that we no longer need
master_df.drop(columns=["month", "year"], inplace=True)

# Move 'tenor' to the first column
cols = master_df.columns.tolist()
cols.insert(0, cols.pop(cols.index("tenor")))
master_df = master_df[cols]

# master_df is the df that will be exported to sheet 1 in the workbook

## ======================== Euro-dollar curve construction (for sheet 2) - Ola's request ========================
# Serves as a potential indicator or early-warning that could affect some of the Geo-spreads

# fetching the info
def fetch_forward_fx_rates(url, tenor_list):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    data = []

    for tenor in tenor_list:
        td = soup.find("td", string=lambda s: s and tenor in s)
        if not td:
            print(f"{tenor} not found.")
            continue

        siblings = td.find_next_siblings("td", limit=3)
        if len(siblings) < 3:
            print(f"Not enough data for {tenor}.")
            continue

        try:
            bid = float(siblings[0].text.strip())
            ask = float(siblings[1].text.strip())
            mid = float(siblings[2].text.strip())
            data.append({
                "tenor": tenor,
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "mid": round(mid, 4)
            })
        except Exception as e:
            print(f"Error parsing {tenor}: {e}")
            continue

    return pd.DataFrame(data)

tenors = [
    "Overnight", "Tomorrow Next", "Spot Next", # short term tenors
    "One Week", "Two Week", "Three Week", # week to 3 week tenors
    "One Month", "Two Month", "Three Month", "Four Month", "Five Month", "Six Month",
    "Seven Month", "Eight Month", "Nine Month", "Ten Month", "Eleven Month",
    "One Year", "Two Year", "Three Year", "Four Year", "Five Year" # mid to back rate refs
]
tenor_eur_dol = tenors + ['Six Year', 'Seven Year', 'Ten Year']

eur_dol_url = "https://www.fxempire.com/currencies/eur-usd/forward-rates"
eurdol_fwd_data = fetch_forward_fx_rates(eur_dol_url, tenor_eur_dol)

## ======================== Import into Excel Workbook ========================

# Create a filename with today's date
today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"cross_bbl_pricing_sheet_{today_str}.xlsx"

# Define RBOB-specific 4dp columns
rbob_cols_4dp = [
    'RBOB flat ($/gal)',
    'RBOB TS ($/gal)',
    'Gasoline ARB ~ rbob v ebob ($/gal)'
]

# Export the DataFrames with formatting
with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    # Write sheets
    master_df.to_excel(writer, sheet_name='Product Fwd curve', index=False)
    eurdol_fwd_data.to_excel(writer, sheet_name='Euro-dollar Fwds', index=False)

    # Access workbook and worksheets
    workbook = writer.book
    sheet1 = writer.sheets['Product Fwd curve']
    sheet2 = writer.sheets['Euro-dollar Fwds']

    # Define formats
    center_format_2dp = workbook.add_format({
        'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'
    })
    center_format_4dp = workbook.add_format({
        'align': 'center', 'valign': 'vcenter', 'num_format': '0.0000'
    })

    # Format and auto-size Product Fwd curve sheet
    for idx, col in enumerate(master_df.columns):
        # Determine max content width in this column
        max_len = max(
            master_df[col].astype(str).map(len).max(),
            len(str(col))
        ) + 2  # add padding

        # Choose format
        cell_format = center_format_4dp if col in rbob_cols_4dp else center_format_2dp

        # Set column width and format
        sheet1.set_column(idx, idx, max_len, cell_format)

    # Format and auto-size Euro-dollar Fwds sheet (all columns 4dp)
    for idx, col in enumerate(eurdol_fwd_data.columns):
        max_len = max(
            eurdol_fwd_data[col].astype(str).map(len).max(),
            len(str(col))
        ) + 2
        sheet2.set_column(idx, idx, max_len, center_format_4dp)

    # Freeze panes
    sheet1.freeze_panes(0, 1)  # Freeze first column
    sheet2.freeze_panes(1, 0)  # Freeze first row

# Serves as a checker in the original ipynb
#print(f"Exported to: {filename}")

## ======================== Delivery mechanism ========================
def send_email_with_attachment(
    filename,          # Excel file path to attach
    subject,           # Subject line for the email
    body,              # Email message body (plain text)
    to_email,          # List of TO recipients
    from_email,        # Sender's email address
    smtp_server,       # SMTP server address (e.g. smtp.gmail.com)
    smtp_port,         # SMTP port (e.g. 465 for SSL)
    login,             # Login/email used to authenticate with SMTP
    password,          # Password or App password for SMTP auth
    cc=None,           # Optional list of CC recipients
    bcc=None           # Optional list of BCC recipients
):
    # Create base email object
    msg = EmailMessage()                                # Create a blank email
    msg['Subject'] = subject                            # Set subject line
    msg['From'] = from_email                            # Set sender
    msg['To'] = ', '.join(to_email)                     # Join TO list into a single string

    if cc:
        msg['Cc'] = ', '.join(cc)                       # Join CC list if provided

    msg.set_content(body)                               # Add the email body (plain text)

    # Attach the Excel file to the email
    with open(filename, 'rb') as f:                     # Open file in binary mode
        msg.add_attachment(
            f.read(),                                   # Read contents of the file
            maintype='application',                     # MIME type: application/octet-stream (generic binary)
            subtype='octet-stream',
            filename=os.path.basename(filename)         # Use only the filename in attachment
        )

    # Combine all recipients (To + Cc + Bcc)
    all_recipients = to_email + (cc or []) + (bcc or [])  # Ensure all recipients get the message

    # Send the message via secure SMTP
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:  # Establish secure SMTP session
        smtp.login(login, password)                         # Log in to SMTP server
        smtp.send_message(msg, to_addrs=all_recipients)     # Send the fully formed message

# Email delivery configuration — ready to call at the end of your script
send_email_with_attachment(
    filename=filename,                                      # Path to Excel file generated earlier
    subject=f"Cross Barrel Pricing Sheet – {today_str}",    # Dynamic subject line with today's date
    body="Hi Ola,\n\nPlease find attached the latest cross-barrel pricing sheet.",  # Plain-text body
    to_email=["Ola.Hansson@irh.ae"],                        # Primary recipient
    cc=["energy@irh.ae", "vedant.bundellu@irh.ae"],# Optional CC list

    #bcc=["hidden1@irh.ae", "hidden2@irh.ae"],  # Commented out for now (uncomment to add more people in the email)

    # Change the sending email address to Vedant by the end of the week
    from_email="vedantxyz1@gmail.com",                   # Sender address
    smtp_server="smtp.gmail.com",                           # Gmail SMTP
    smtp_port=465,                                          # SSL port for Gmail
    login="vedantxyz1@gmail.com",                        # Same as sender
    password=os.environ.get("EMAIL_PASSWORD")               # Pull password securely from environment variable
)

