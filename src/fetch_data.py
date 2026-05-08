import glob
import io
import json
import logging
import os
import socket
import sys
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import yfinance as yf
import pandas as pd

# Process-wide socket timeout. yfinance doesn't expose a timeout parameter
# on yf.download / yf.Ticker(...).info / .get_earnings_dates / .history,
# and an unbounded HTTPS connection can wedge the whole fetch script if
# Yahoo's backend stalls on a single ticker (observed at sp1500 scale).
# 10s is generous for a healthy yfinance call (typical: <2s) and short
# enough that a real hang surfaces as a per-ticker exception, letting
# fetch_sp1500's retry/sanity paths take over. Setting the SOCKET default
# rather than wrapping each call: yfinance speaks HTTPS through urllib3,
# which honors socket.setdefaulttimeout as the fallback. Existing explicit
# timeouts (e.g. finnhub_insider_signals.py REQUEST_TIMEOUT) still take
# precedence — this is a fallback for the cases that didn't have one.
socket.setdefaulttimeout(10.0)

try:
    import pyarrow  # noqa: F401  (required engine for parquet I/O)
    _HAS_PYARROW = True
except ImportError:
    print("  [CACHE] pyarrow not installed — caching disabled. "
          "Run `pip install pyarrow` to enable fast incremental downloads.")
    _HAS_PYARROW = False


# --- Verbosity + yfinance noise suppression --------------------------------
# yfinance prints per-ticker error messages like "1 Failed download: ['XYZ']"
# and "$XYZ: possibly delisted" on every failed lookup. With a 491-ticker
# universe and intermittent failures, that floods stdout with hundreds of
# lines. We swallow them by default and emit one summary line at the end.
# Dashboard callers can opt back in by setting fetch_data.VERBOSE = True
# (wired to the --verbose CLI flag).
VERBOSE = False


def set_verbose(v: bool) -> None:
    """Toggle per-ticker failure messages from yfinance."""
    global VERBOSE
    VERBOSE = bool(v)


@contextmanager
def _quiet_yfinance():
    """Silence yfinance's stdout/stderr chatter for the duration of a call.

    When VERBOSE is True this is a no-op so users can still debug individual
    ticker failures via ``--verbose``.
    """
    if VERBOSE:
        yield
        return
    # Global log mute catches every yfinance sublogger (names are added
    # dynamically). sys.stdout/stderr redirection catches print() calls in
    # the main thread. Direct fd redirect via os.dup2 is what catches writes
    # from yfinance's internal ThreadPoolExecutor — sys.stdout replacement
    # alone doesn't propagate into worker threads on Windows.
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    devnull_fd    = os.open(os.devnull, os.O_WRONLY)
    saved_out_fd  = os.dup(1)
    saved_err_fd  = os.dup(2)
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved_out_fd, 1)
            os.dup2(saved_err_fd, 2)
    finally:
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        os.close(devnull_fd)
        logging.disable(prev_disable)

NASDAQ_100_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
    "AMGN", "AMZN", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB",
    "BKNG", "BKR", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COIN", "COST",
    "CPRT", "CRWD", "CSCO", "CSGP", "CTAS", "CTSH", "DASH", "DDOG", "DLTR",
    "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GILD", "GOOG",
    "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC",
    "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI",
    "META", "MNST", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL",
    "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL",
    "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL",
    "ZS",
]

SECTOR_MAP = {
    # Technology
    "AAPL": "tech", "ADBE": "tech", "ADI": "tech", "ADSK": "tech",
    "AMAT": "tech", "AMD": "tech", "APP": "tech",
    "ARM": "tech", "ASML": "tech", "AVGO": "tech", "CDNS": "tech",
    "CRWD": "tech", "CSCO": "tech", "CSGP": "tech", "CTSH": "tech",
    "DDOG": "tech", "EA": "tech", "FTNT": "tech", "GOOG": "tech",
    "INTC": "tech", "INTU": "tech", "KLAC": "tech",
    "LRCX": "tech", "MDB": "tech", "META": "tech", "MRVL": "tech",
    "MSFT": "tech", "MU": "tech", "NFLX": "tech", "NVDA": "tech",
    "NXPI": "tech", "ON": "tech", "PANW": "tech", "PLTR": "tech",
    "QCOM": "tech", "SMCI": "tech", "SNPS": "tech", "TEAM": "tech",
    "TTD": "tech", "TXN": "tech", "WDAY": "tech", "ZS": "tech",
    "CDW": "tech", "MCHP": "tech",
    # Consumer
    "ABNB": "consumer", "AMZN": "consumer", "BKNG": "consumer",
    "CHTR": "consumer", "CMCSA": "consumer", "COST": "consumer",
    "CPRT": "consumer", "DASH": "consumer", "DLTR": "consumer",
    "KDP": "consumer", "KHC": "consumer", "LULU": "consumer",
    "MAR": "consumer", "MDLZ": "consumer", "MELI": "consumer",
    "MNST": "consumer", "ODFL": "consumer", "ORLY": "consumer",
    "PCAR": "consumer", "PDD": "consumer", "PEP": "consumer",
    "ROST": "consumer", "SBUX": "consumer", "TSLA": "consumer",
    "TTWO": "consumer", "WBD": "consumer",
    # Healthcare
    "AMGN": "healthcare", "AZN": "healthcare", "BIIB": "healthcare",
    "DXCM": "healthcare", "GEHC": "healthcare", "GILD": "healthcare",
    "IDXX": "healthcare", "ILMN": "healthcare", "ISRG": "healthcare",
    "REGN": "healthcare", "VRTX": "healthcare",
    # Financial
    "ADP": "financial", "COIN": "financial", "FAST": "financial",
    "PAYX": "financial", "PYPL": "financial", "ROP": "financial",
    "VRSK": "financial",
    # Industrial / Utilities / Energy
    "AEP": "industrial", "BKR": "industrial", "CEG": "industrial",
    "CTAS": "industrial", "EXC": "industrial", "FANG": "industrial",
    "HON": "industrial", "LIN": "industrial", "TMUS": "industrial",
    "XEL": "industrial",
}


SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APP",
    "APTV", "ARE", "ATO", "AVGO", "AVY", "AWK", "AXP", "AZO",
    "BA", "BAC", "BAX", "BBWI", "BBY", "BDX", "BEN", "BG", "BIIB",
    "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BWA", "BX", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
    "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
    "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL",
    "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH",
    "CTVA", "CVS", "CVX",
    "D", "DAL", "DASH", "DD", "DE", "DECK", "DG", "DGX", "DHI", "DHR",
    "DIS", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
    "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH",
    "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "EVRG",
    "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV",
    "FICO", "FIS", "FISV", "FITB", "FMC", "FOX", "FOXA", "FRT",
    "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOG", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX", "HON", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU",
    "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR",
    "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PDD", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR",
    "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA",
    "PSX", "PTC", "PVH", "PWR", "PYPL",
    "QCOM", "QRVO",
    "RCL", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX",
    "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ",
    "SWK", "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
    "VTR", "VTRS", "VZ",
    "WAB", "WAT", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
    "WMT", "WRB", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XRAY", "XYL",
    "YUM",
    "ZBH", "ZBRA", "ZTS",
]


# Combined universe: Nasdaq 100 + S&P 500, deduplicated.
# Existing studies (#325, all v1/v2/v3 dashboards) trained against this set;
# preserved verbatim so they stay reproducible against pre_v2_20260505.
UNIVERSE_TICKERS = sorted(set(NASDAQ_100_TICKERS + SP500_TICKERS))


# --- S&P 1500 expansion (S&P 400 MidCap + S&P 600 SmallCap) ----------------
# Sourced from Wikipedia's constituent tables on 2026-05-08 via
# src/fetch_constituents.py — re-run that script to refresh. Tickers are
# yfinance-normalized (BRK.B → BRK-B). Some symbols overlap SP500_TICKERS
# (recent index moves); SP1500_TICKERS dedupes the union.

SP400_TICKERS = [
    "AA", "AAL", "AAON", "ACI", "ACM", "ADC", "AEIS", "AFG", "AGCO", "AHR",
    "AIT", "ALGM", "ALK", "ALLY", "ALV", "AM", "AMG", "AMH", "AMKR", "AN",
    "ANF", "APG", "APPF", "AR", "ARMK", "ARW", "ARWR", "ASB", "ASH", "ATI",
    "ATR", "AVAV", "AVNT", "AVT", "AVTR", "AXTA", "AYI", "BAH", "BBWI",
    "BC", "BCO", "BDC", "BHF", "BILL", "BIO", "BJ", "BKH", "BLD", "BLKB",
    "BMRN", "BRBR", "BRKR", "BROS", "BRX", "BSY", "BURL", "BWA", "BWXT",
    "BYD", "CACI", "CAR", "CART", "CAVA", "CBSH", "CBT", "CCK", "CDP",
    "CELH", "CFR", "CG", "CGNX", "CHDN", "CHE", "CHH", "CHRD", "CHWY",
    "CLF", "CLH", "CMC", "CNH", "CNM", "CNO", "CNX", "CNXC", "COKE",
    "COLB", "COLM", "COTY", "CPRI", "CR", "CRBG", "CROX", "CRS", "CRUS",
    "CSL", "CTRE", "CUBE", "CUZ", "CVLT", "CW", "CXT", "CYTK", "DAR",
    "DBX", "DCI", "DINO", "DKS", "DLB", "DOCN", "DOCS", "DOCU", "DT",
    "DTM", "DUOL", "DY", "EEFT", "EGP", "EHC", "ELAN", "ELF", "ELS", "ENS",
    "ENSG", "ENTG", "EPR", "EQH", "ESAB", "ESNT", "EVR", "EWBC", "EXEL",
    "EXLS", "EXP", "EXPO", "FAF", "FBIN", "FCFS", "FCN", "FFIN", "FHI",
    "FHN", "FIVE", "FLEX", "FLG", "FLO", "FLR", "FLS", "FN", "FNB", "FND",
    "FNF", "FOUR", "FR", "FTI", "G", "GAP", "GATX", "GBCI", "GEF", "GGG",
    "GHC", "GLPI", "GME", "GMED", "GNTX", "GPK", "GT", "GTLS", "GWRE",
    "GXO", "H", "HAE", "HALO", "HGV", "HIMS", "HL", "HLI", "HLNE", "HOG",
    "HOMB", "HQY", "HR", "HRB", "HWC", "HXL", "IBOC", "IDA", "IDCC",
    "ILMN", "INGR", "IPGP", "IRT", "ITT", "JAZZ", "JEF", "JHG", "JLL",
    "KBH", "KBR", "KD", "KEX", "KNF", "KNSL", "KNX", "KRC", "KRG", "KTOS",
    "LAD", "LAMR", "LEA", "LECO", "LFUS", "LIVN", "LNTH", "LOPE", "LPX",
    "LSCC", "LSTR", "M", "MANH", "MASI", "MAT", "MEDP", "MIDD", "MKSI",
    "MLI", "MMS", "MOG-A", "MORN", "MP", "MSA", "MSM", "MTDR", "MTG",
    "MTN", "MTSI", "MTZ", "MUR", "MUSA", "MZTI", "NBIX", "NEU", "NFG",
    "NJR", "NLY", "NNN", "NOV", "NOVT", "NSA", "NTNX", "NVST", "NVT",
    "NWE", "NXST", "NXT", "NYT", "OC", "OGE", "OGS", "OHI", "OKTA", "OLED",
    "OLLI", "OLN", "ONB", "ONTO", "OPCH", "ORA", "ORI", "OSK", "OVV",
    "OZK", "PAG", "PATH", "PB", "PBF", "PCTY", "PEGA", "PEN", "PFGC",
    "PII", "PINS", "PK", "PLNT", "PNFP", "POR", "POST", "PPC", "PR", "PRI",
    "PSN", "PSTG", "PVH", "QLYS", "R", "RBA", "RBC", "REXR", "RGA", "RGEN",
    "RGLD", "RH", "RLI", "RMBS", "RNR", "ROIV", "RPM", "RRC", "RRX", "RS",
    "RYAN", "RYN", "SAIA", "SAIC", "SAM", "SARO", "SBRA", "SCI", "SEIC",
    "SF", "SFM", "SGI", "SHC", "SIGI", "SITM", "SLAB", "SLGN", "SLM",
    "SMG", "SNX", "SOLS", "SON", "SPXC", "SR", "SSB", "SSD", "ST", "STAG",
    "STRL", "STWD", "SWX", "SYNA", "TCBI", "TEX", "THC", "THG", "THO",
    "TKR", "TLN", "TMHC", "TNL", "TOL", "TREX", "TRU", "TTC", "TTEK",
    "TTMI", "TWLO", "TXNM", "TXRH", "UBSI", "UFPI", "UGI", "ULS", "UMBF",
    "UNM", "USFD", "UTHR", "VAL", "VC", "VFC", "VICR", "VLY", "VMI", "VNO",
    "VNOM", "VNT", "VOYA", "VVV", "WAL", "WBS", "WCC", "WEX", "WFRD", "WH",
    "WHR", "WING", "WLK", "WMG", "WMS", "WPC", "WSO", "WTFC", "WTRG",
    "WTS", "WWD", "XPO", "XRAY", "YETI", "ZION",
]


SP600_TICKERS = [
    "AAMI", "AAP", "AAT", "ABCB", "ABG", "ABM", "ABR", "ACA", "ACAD",
    "ACHC", "ACIW", "ACLS", "ACMR", "ACT", "ADAM", "ADEA", "ADMA", "ADNT",
    "ADT", "ADUS", "AEO", "AESI", "AGO", "AGX", "AGYS", "AHCO", "AIN",
    "AIR", "AKR", "ALG", "ALGT", "ALKS", "ALRM", "AMN", "AMPH", "AMR",
    "AMRX", "AMSF", "AMTM", "AMWD", "ANDE", "ANIP", "AORT", "AOSL", "APAM",
    "APLE", "APLS", "APOG", "ARCB", "ARI", "ARLO", "AROC", "ARR", "ASGN",
    "ASO", "ASTE", "ASTH", "ATEN", "ATMU", "AUB", "AVA", "AVNS", "AWI",
    "AWR", "AX", "AZTA", "AZZ", "BANC", "BANF", "BANR", "BBT", "BCC",
    "BCPC", "BFH", "BFS", "BGC", "BHE", "BJRI", "BKE", "BKU", "BL", "BLFS",
    "BMI", "BNL", "BOH", "BOOT", "BOX", "BRC", "BTSG", "BTU", "BXMT",
    "CABO", "CAKE", "CALM", "CALX", "CALY", "CARG", "CASH", "CATY", "CBRL",
    "CBU", "CC", "CCOI", "CCS", "CE", "CENT", "CENTA", "CENX", "CERT",
    "CFFN", "CHCO", "CHEF", "CLB", "CLSK", "CNK", "CNMD", "CNR", "CNS",
    "CNXN", "COCO", "COHU", "COLL", "CON", "CORT", "CPF", "CPK", "CPRX",
    "CRC", "CRGY", "CRI", "CRK", "CRSR", "CRVL", "CSGS", "CSR", "CSW",
    "CTKB", "CTS", "CUBI", "CURB", "CVBF", "CVCO", "CVI", "CVSA", "CWEN",
    "CWEN-A", "CWK", "CWST", "CWT", "CXM", "CXW", "CZR", "DAN", "DBD",
    "DCH", "DCOM", "DEA", "DEI", "DFH", "DFIN", "DGII", "DIOD", "DLX",
    "DNOW", "DORM", "DRH", "DV", "DXC", "DXPE", "EAT", "ECG", "ECPG",
    "EFC", "EGBN", "EIG", "EMBC", "EMN", "ENOV", "ENPH", "ENR", "ENVA",
    "EPAC", "EPC", "EPRT", "ESE", "ESI", "ETD", "ETSY", "EVTC", "EXPI",
    "EXTR", "EYE", "EZPW", "FBK", "FBNC", "FBP", "FBRT", "FCF", "FCPT",
    "FDP", "FELE", "FFBC", "FHB", "FIBK", "FIZZ", "FMC", "FORM", "FOXF",
    "FRPT", "FSS", "FTDR", "FTRE", "FUL", "FULT", "FUN", "FWRD", "GBX",
    "GDYN", "GEO", "GFF", "GIII", "GKOS", "GNL", "GNW", "GO", "GOGO",
    "GOLF", "GPI", "GRBK", "GSHD", "GTES", "GTM", "GTY", "GVA", "HAFC",
    "HASI", "HAYW", "HCC", "HCI", "HCSG", "HE", "HFWA", "HIW", "HLIT",
    "HLX", "HMN", "HNI", "HOPE", "HP", "HRMY", "HSTM", "HTH", "HTLD",
    "HTO", "HTZ", "HUBG", "HWKN", "HZO", "IAC", "IART", "IBP", "ICHR",
    "ICUI", "IIIN", "IIPR", "INDB", "INDV", "INSP", "INSW", "INVA", "INVX",
    "IOSP", "IPAR", "IRDM", "ITGR", "ITRI", "JBGS", "JBLU", "JBSS", "JBTM",
    "JJSF", "JOE", "JXN", "KAI", "KALU", "KFY", "KGS", "KLIC", "KMPR",
    "KMT", "KMX", "KN", "KNTK", "KOP", "KRYS", "KSS", "KTB", "KW", "KWR",
    "LAUR", "LBRT", "LCII", "LEG", "LFST", "LGIH", "LGND", "LIF", "LKFN",
    "LKQ", "LMAT", "LNC", "LNN", "LPG", "LQDT", "LRN", "LTC", "LTH",
    "LUMN", "LW", "LXP", "LYFT", "LZ", "LZB", "MAC", "MAN", "MARA", "MATW",
    "MATX", "MBC", "MBIN", "MC", "MCRI", "MCW", "MCY", "MD", "MDU", "MGEE",
    "MGY", "MHK", "MHO", "MIR", "MKTX", "MLKN", "MMI", "MMSI", "MNRO",
    "MOH", "MPT", "MRCY", "MRP", "MRTN", "MSEX", "MSGS", "MTCH", "MTH",
    "MTRN", "MTUS", "MTX", "MWA", "MXL", "MYRG", "NABL", "NATL", "NAVI",
    "NBHC", "NBTB", "NE", "NEO", "NEOG", "NGVT", "NHC", "NMIH", "NOG",
    "NPK", "NPO", "NSIT", "NSP", "NSSC", "NTCT", "NVRI", "NWBI", "NWL",
    "NWN", "NX", "NXRT", "OFG", "OGN", "OI", "OII", "OMCL", "OPLN", "OSIS",
    "OSW", "OTTR", "OUT", "OXM", "PAHC", "PARR", "PATK", "PAYC", "PAYO",
    "PBH", "PBI", "PCRX", "PDFS", "PEB", "PECO", "PENG", "PENN", "PFBC",
    "PFS", "PGNY", "PHIN", "PI", "PIPR", "PJT", "PLAB", "PLMR", "PLUS",
    "PLXS", "PMT", "POWI", "POWL", "PRA", "PRAA", "PRDO", "PRG", "PRGO",
    "PRGS", "PRIM", "PRK", "PRKS", "PRLB", "PRSU", "PRVA", "PSMT", "PTCT",
    "PTEN", "PTGX", "PZZA", "QDEL", "QNST", "QRVO", "QTWO", "RAL", "RAMP",
    "RCUS", "RDN", "RDNT", "RES", "REX", "REYN", "REZI", "RHI", "RHP",
    "RITM", "RNG", "RNST", "ROCK", "ROG", "RRR", "RUN", "RUSHA", "RWT",
    "RXO", "SABR", "SAFE", "SAFT", "SAH", "SANM", "SBCF", "SBH", "SBSI",
    "SCHL", "SCL", "SCSC", "SDGR", "SEDG", "SEM", "SEZL", "SFBS", "SFNC",
    "SHAK", "SHEN", "SHO", "SHOO", "SIG", "SKT", "SKY", "SKYW", "SLG",
    "SLVM", "SM", "SMP", "SMPL", "SMTC", "SNCY", "SNDR", "SNEX", "SONO",
    "SPHR", "SPNT", "SPSC", "SRPT", "SSTK", "STAA", "STBA", "STC", "STEL",
    "STEP", "STRA", "SUPN", "SXI", "SXT", "TALO", "TBBK", "TDC", "TDS",
    "TDW", "TFIN", "TFX", "TGTX", "THRM", "TILE", "TMDX", "TMP", "TNC",
    "TNDM", "TPH", "TR", "TRIP", "TRMK", "TRN", "TRNO", "TRST", "TRUP",
    "TWO", "UA", "UAA", "UCB", "UCTT", "UE", "UFCS", "UFPT", "UHT", "UNF",
    "UNFI", "UNIT", "UPBD", "UPWK", "URBN", "USPH", "UTL", "UVV", "VAC",
    "VCEL", "VCTR", "VCYT", "VECO", "VGNT", "VIAV", "VIR", "VIRT", "VITL",
    "VRE", "VRRM", "VRTS", "VSAT", "VSCO", "VSEC", "VSH", "VSNT", "VSTS",
    "VTOL", "VYX", "WABC", "WAFD", "WAY", "WD", "WDFC", "WEN", "WERN",
    "WGO", "WHD", "WINA", "WKC", "WLY", "WOR", "WRLD", "WS", "WSC", "WSFS",
    "WSR", "WT", "WU", "WWW", "XHR", "XNCR", "XPEL", "YELP", "YOU", "ZD",
    "ZWS",
]


# S&P 1500 = NASDAQ 100 ∪ S&P 500 ∪ S&P 400 ∪ S&P 600 (deduped). Used by
# the new sp1500 universe option. Existing 490-ticker scripts keep using
# UNIVERSE_TICKERS unchanged for reproducibility against pre_v2_20260505.
SP1500_TICKERS = sorted(set(
    NASDAQ_100_TICKERS + SP500_TICKERS + SP400_TICKERS + SP600_TICKERS))


# --- Cache snapshot system (PAPER_TRADER_DATA_ROOT) ----------------------
_DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models"))
DATA_ROOT = os.environ.get("PAPER_TRADER_DATA_ROOT", _DEFAULT_DATA_ROOT)
SNAPSHOT_MODE = os.environ.get("PAPER_TRADER_DATA_ROOT") is not None

SECTOR_MAP_CACHE    = os.path.join(DATA_ROOT, "cache", "sector_map.json")
SECTOR_MAP_TTL_DAYS = 30
_SECTOR_MAP_FLUSH_EVERY = 25  # checkpoint cache every N yfinance lookups


def _snapshot_miss(cache_name: str, path: str, required: str) -> "RuntimeError":
    return RuntimeError(
        f"[CACHE_SNAPSHOT_MISS] Cache miss in snapshot mode.\n"
        f"  Cache: {cache_name}\n"
        f"  Path: {path}\n"
        f"  Snapshot: {os.environ.get('PAPER_TRADER_DATA_ROOT')}\n"
        f"  Required: {required}\n"
        f"  Action: re-create snapshot with current data, or fix snapshot to "
        f"include this data."
    )


def _write_sector_cache(result: dict[str, str]) -> bool:
    """Write the full merged sector map to disk via a .tmp rename so a mid-
    write crash (or a bad key type) can't corrupt the real cache file.
    Returns True on success."""
    # Skip any non-string keys/values defensively — sort_keys=True chokes on
    # a None ticker and would leave the cache empty otherwise.
    clean = {k: v for k, v in result.items()
             if isinstance(k, str) and isinstance(v, str)}
    tmp = SECTOR_MAP_CACHE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, sort_keys=True)
        os.replace(tmp, SECTOR_MAP_CACHE)
        return True
    except Exception as e:
        print(f"  [CACHE] Failed to write sector map ({e})")
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        return False


def build_sector_map(tickers: list[str]) -> dict[str, str]:
    """Extend SECTOR_MAP with yfinance lookups for unmapped tickers.

    The full merged map (static ``SECTOR_MAP`` entries + every yfinance
    lookup ever done) is cached to ``sector_map.json`` with a 30-day TTL —
    sectors almost never change and per-ticker ``yf.Ticker().info`` calls
    are slow. Subsequent runs skip the network entirely as long as every
    requested ticker is already in the cache.

    The cache is flushed every 25 lookups so a Ctrl-C during the first
    population doesn't wipe out progress.
    """
    sector_lookup = {
        "Technology": "tech", "Communication Services": "tech",
        "Consumer Cyclical": "consumer", "Consumer Defensive": "consumer",
        "Healthcare": "healthcare",
        "Financial Services": "financial",
        "Industrials": "industrial", "Utilities": "industrial",
        "Energy": "industrial", "Basic Materials": "industrial",
        "Real Estate": "financial",
    }
    os.makedirs(os.path.dirname(SECTOR_MAP_CACHE), exist_ok=True)

    # Load cache if present and fresh.
    cached: dict[str, str] = {}
    cache_age_days: float | None = None
    if os.path.exists(SECTOR_MAP_CACHE):
        cache_age_days = (time.time() - os.path.getmtime(SECTOR_MAP_CACHE)) / 86400.0
        if cache_age_days < SECTOR_MAP_TTL_DAYS:
            try:
                with open(SECTOR_MAP_CACHE, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"  [CACHE] Sector map read from {SECTOR_MAP_CACHE} "
                      f"({len(cached)} entries, {cache_age_days:.1f} days old)")
            except Exception as e:
                print(f"  [CACHE] Sector map unreadable ({e}) - rebuilding")
                cached = {}
        else:
            print(f"  [CACHE] Sector map stale ({cache_age_days:.0f} days old) "
                  f"- rebuilding")
            cached = {}

    # Start from static map, then overlay the cache (cache wins on conflict
    # — it reflects the most recent yfinance classification).
    result: dict[str, str] = dict(SECTOR_MAP)
    result.update(cached)

    unmapped = [t for t in tickers if t not in result]
    n_loaded = len(tickers) - len(unmapped)

    if not unmapped:
        print(f"  [CACHE] Sector map: all {len(tickers)} tickers loaded from "
              f"disk (0 yfinance lookups)")
        return result

    if SNAPSHOT_MODE:
        raise _snapshot_miss(
            "sector_map", SECTOR_MAP_CACHE,
            f"all {len(tickers)} tickers present in cache "
            f"(missing {len(unmapped)}, e.g. {unmapped[:3]})")

    print(f"  [CACHE] Sector map: {n_loaded} loaded from disk, "
          f"{len(unmapped)} looked up fresh")

    wrote_ok = False
    for i, tkr in enumerate(unmapped, 1):
        try:
            with _quiet_yfinance():
                sector = yf.Ticker(tkr).info.get("sector", "")
            result[tkr] = sector_lookup.get(sector, "other")
        except Exception:
            result[tkr] = "other"

        # Incremental flush so a crash mid-loop doesn't lose all progress.
        if i % _SECTOR_MAP_FLUSH_EVERY == 0 or i == len(unmapped):
            wrote_ok = _write_sector_cache(result)

    if wrote_ok:
        print(f"  [CACHE] Sector map written to {SECTOR_MAP_CACHE} "
              f"({len(result)} total entries)")
    return result


def get_stock_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for a list of tickers via yfinance.

    Args:
        tickers: List of US stock ticker symbols.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        Dictionary mapping each ticker to its cleaned OHLCV DataFrame.
    """
    with _quiet_yfinance():
        raw = yf.download(tickers, start=start, end=end,
                          group_by="ticker", auto_adjust=True, progress=False)

    result = {}
    failures: list[str] = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            df = df.ffill().dropna()
            if df.empty:
                failures.append(ticker)
                if VERBOSE:
                    print(f"  [SKIP] {ticker}: no data after cleaning")
                continue
            result[ticker] = df
        except Exception as e:
            failures.append(ticker)
            if VERBOSE:
                print(f"  [SKIP] {ticker}: {e}")

    if failures and not VERBOSE:
        print(f"  [CACHE] {len(failures)} tickers failed to download "
              f"(possibly delisted or rate limited)")
    return result


def _cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"{ticker}.parquet")


def _load_cached(path: str) -> pd.DataFrame | None:
    """Load a single cached parquet, return None if missing/corrupt."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty or "Close" not in df.columns:
            return None
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def _download_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download one ticker's OHLCV, cleaned, or None on failure.

    yfinance chatter is silenced unless ``VERBOSE`` is set.
    """
    try:
        with _quiet_yfinance():
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.ffill().dropna()
        return df if not df.empty else None
    except Exception:
        return None


def get_stock_data_cached(tickers: list[str], start: str, end: str,
                          cache_dir: str) -> dict[str, pd.DataFrame]:
    """Drop-in replacement for get_stock_data() that caches each ticker's
    OHLCV as a separate parquet file under ``cache_dir`` and only fetches
    the incremental days that aren't already on disk.

    If pyarrow isn't available, silently falls back to :func:`get_stock_data`
    (no cache). If a single ticker's cache file is missing or corrupted,
    that one ticker falls back to a full download while the rest still use
    the cache.
    """
    if not _HAS_PYARROW:
        return get_stock_data(tickers, start, end)

    os.makedirs(cache_dir, exist_ok=True)
    end_ts = pd.Timestamp(end)
    start_ts = pd.Timestamp(start)

    result: dict[str, pd.DataFrame] = {}
    n_from_disk = 0
    n_fresh     = 0
    n_appended  = 0
    max_appended_days = 0
    update_failures: list[str] = []    # had a cache but fresh fetch returned nothing
    fresh_failures:  list[str] = []    # no cache and no fetch data
    skipped:         list[str] = []    # ended up with no usable data window

    for ticker in tickers:
        path = _cache_path(cache_dir, ticker)
        cached = _load_cached(path)

        if cached is None:
            if SNAPSHOT_MODE:
                raise _snapshot_miss(
                    f"price_cache[{ticker}]", path,
                    "per-ticker parquet present and readable")
            df = _download_single(ticker, start, end)
            if df is None or df.empty:
                fresh_failures.append(ticker)
                continue
            try:
                df.to_parquet(path)
            except Exception as e:
                if VERBOSE:
                    print(f"  [CACHE] {ticker}: failed to write parquet ({e})")
            result[ticker] = df.loc[start_ts:end_ts]
            n_fresh += 1
            continue

        # Cache hit. Backfill earlier history if needed (silently on fail).
        # In snapshot mode, skip backfill entirely — the snapshot's cache is
        # authoritative. Minor boundary gaps (weekends, holidays where the
        # requested start lands on a non-trading day) are not real misses.
        if cached.index.min() > start_ts and not SNAPSHOT_MODE:
            backfill_end = (cached.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_old = _download_single(ticker, start, backfill_end)
            if new_old is not None and not new_old.empty:
                cached = pd.concat([new_old, cached])
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()

        # Incremental forward fetch. Same snapshot-mode skip as backfill.
        last_date = cached.index.max()
        next_date = (last_date + pd.Timedelta(days=1)).normalize()
        appended_days = 0
        if next_date <= end_ts and not SNAPSHOT_MODE:
            new_df = _download_single(ticker,
                                      next_date.strftime("%Y-%m-%d"),
                                      end)
            if new_df is None or new_df.empty:
                # We thought there should be new data but got nothing —
                # counts as a failed update. Not fatal: still use the cache.
                update_failures.append(ticker)
            else:
                new_df = new_df[new_df.index > last_date]
                if not new_df.empty:
                    cached = pd.concat([cached, new_df])
                    cached = cached[~cached.index.duplicated(keep="last")].sort_index()
                    appended_days = len(new_df)

        if appended_days > 0:
            try:
                cached.to_parquet(path)
            except Exception as e:
                if VERBOSE:
                    print(f"  [CACHE] {ticker}: failed to update parquet ({e})")
            n_appended += 1
            max_appended_days = max(max_appended_days, appended_days)

        sliced = cached.loc[start_ts:end_ts]
        if sliced.empty:
            skipped.append(ticker)
            continue
        result[ticker] = sliced
        n_from_disk += 1

    print(f"  Cache: {n_from_disk} tickers loaded from disk, "
          f"{n_fresh} tickers downloaded fresh, "
          f"{n_appended} tickers appended with up to {max_appended_days} new trading days")

    # Single summary line for all the failures (vs hundreds of individual ones).
    total_failed = len(update_failures) + len(fresh_failures) + len(skipped)
    if total_failed:
        if VERBOSE:
            if fresh_failures:
                print(f"  [CACHE] fresh download failed: {', '.join(fresh_failures)}")
            if update_failures:
                print(f"  [CACHE] incremental update failed: {', '.join(update_failures)}")
            if skipped:
                print(f"  [CACHE] no data in window: {', '.join(skipped)}")
        else:
            print(f"  [CACHE] {total_failed} tickers failed to update "
                  f"(possibly delisted or rate limited) — using cached data")
    return result


def clear_cache(cache_dir: str) -> int:
    """Delete every parquet file in ``cache_dir``. Returns the file count."""
    if not os.path.isdir(cache_dir):
        return 0
    paths = glob.glob(os.path.join(cache_dir, "*.parquet"))
    for p in paths:
        try:
            os.remove(p)
        except OSError as e:
            print(f"  [CACHE] could not delete {p}: {e}")
    print(f"  [CACHE] cleared {len(paths)} parquet files from {cache_dir}")
    return len(paths)


if __name__ == "__main__":
    from datetime import datetime, timedelta

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print(f"Downloading {len(NASDAQ_100_TICKERS)} Nasdaq 100 tickers ({start} to {end})...")
    data = get_stock_data(NASDAQ_100_TICKERS, start, end)
    print(f"\nSuccessfully loaded {len(data)} / {len(NASDAQ_100_TICKERS)} tickers")

    for ticker, df in list(data.items())[:3]:
        print(f"\n{'='*60}")
        print(f"  {ticker}  —  {len(df)} trading days")
        print(f"{'='*60}")
        print(df.head())
