import yfinance as yf
import anthropic
import json
import numpy as np
from typing import Optional

MARKET_SUFFIXES = {
    'NZX': '.NZ',
    'ASX': '.AX',
    'US': '',
    'NYSE': '',
    'NASDAQ': '',
}

def get_yf_ticker(ticker: str, market: str) -> str:
    suffix = MARKET_SUFFIXES.get(market.upper(), '')
    return f"{ticker}{suffix}"

def compute_rsi(prices, period=14):
    prices = np.array(prices, dtype=float)
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

# ── CAGR helpers ──────────────────────────────────────────────────────────────

def _safe_cagr(values: list) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not np.isnan(float(v)) and float(v) > 0]
    if len(vals) < 2:
        return None
    n = len(vals) - 1
    try:
        return round(((vals[-1] / vals[0]) ** (1 / n) - 1) * 100, 2)
    except Exception:
        return None

def _stmt_row(df, keys: list) -> Optional[list]:
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            return df.loc[key].sort_index().values.tolist()
    return None

def compute_revenue_cagr(financials) -> Optional[float]:
    vals = _stmt_row(financials, ['Total Revenue', 'Revenue'])
    return _safe_cagr(vals) if vals else None

def compute_earnings_cagr(financials) -> Optional[float]:
    vals = _stmt_row(financials, [
        'Net Income', 'Net Income Common Stockholders',
        'Net Income Applicable To Common Shares',
    ])
    return _safe_cagr(vals) if vals else None

def compute_fcf_cagr(cashflow) -> Optional[float]:
    if cashflow is None or cashflow.empty:
        return None
    try:
        op = _stmt_row(cashflow, [
            'Operating Cash Flow', 'Total Cash From Operating Activities',
            'Cash Flow From Continuing Operating Activities',
        ])
        if not op:
            return None
        capex = _stmt_row(cashflow, [
            'Capital Expenditure', 'Capital Expenditures',
            'Purchase Of Property Plant And Equipment',
        ])
        fcf = [o + c for o, c in zip(op, capex)] if capex else op
        return _safe_cagr(fcf)
    except Exception:
        return None

def compute_dividend_growth(dividends) -> Optional[float]:
    if dividends is None or len(dividends) < 4:
        return None
    try:
        annual = dividends.resample('YE').sum()
        if len(annual) < 2:
            return None
        vals = [v for v in annual.values.tolist() if v > 0]
        return _safe_cagr(vals)
    except Exception:
        return None

# ── Quantitative score components ─────────────────────────────────────────────

def score_momentum(signals: dict) -> int:
    """Price momentum signals → 0-20 pts."""
    s = 0

    # 3-month return (0-6)
    r3m = signals.get('ret_3mo')
    if r3m is not None:
        if r3m > 15:   s += 6
        elif r3m > 8:  s += 5
        elif r3m > 3:  s += 4
        elif r3m > 0:  s += 3
        elif r3m > -5: s += 1
    else:
        s += 3  # neutral

    # 1-month return via ret_20d (0-4)
    r1m = signals.get('ret_20d')
    if r1m is not None:
        if r1m > 5:    s += 4
        elif r1m > 2:  s += 3
        elif r1m > 0:  s += 2
        elif r1m > -3: s += 1
    else:
        s += 2

    # RSI — reward trending but not overbought (0-4)
    rsi = signals.get('rsi_14')
    if rsi is not None:
        if 50 <= rsi <= 65:                        s += 4
        elif 45 <= rsi < 50 or 65 < rsi <= 70:    s += 3
        elif 40 <= rsi < 45 or 70 < rsi <= 75:    s += 2
        elif 35 <= rsi < 40:                       s += 1
        # <35 or >75: overbought/oversold = 0
    else:
        s += 2

    # Price vs MA50 (0-3)
    vs50 = signals.get('price_vs_ma50_pct')
    if vs50 is not None:
        if vs50 > 5:    s += 3
        elif vs50 > 0:  s += 2
        elif vs50 > -5: s += 1
    else:
        vs20 = signals.get('price_vs_ma20_pct')
        s += (2 if vs20 and vs20 > 0 else 1)

    # Distance from 52-week high (0-3)
    h52 = signals.get('pct_from_52w_high')
    if h52 is not None:
        if h52 > -10:   s += 3
        elif h52 > -20: s += 2
        elif h52 > -35: s += 1
    else:
        s += 1

    return min(s, 20)


def score_growth(info: dict, financials, cashflow) -> tuple[int, dict]:
    """Revenue / earnings / FCF growth → 0-25 pts."""
    s = 0
    computed = {}

    # Revenue CAGR (0-8)
    rev_cagr = compute_revenue_cagr(financials)
    if rev_cagr is not None:
        computed['revenue_cagr_pct'] = rev_cagr
        if rev_cagr > 20:   s += 8
        elif rev_cagr > 15: s += 7
        elif rev_cagr > 10: s += 6
        elif rev_cagr > 5:  s += 5
        elif rev_cagr > 2:  s += 3
        elif rev_cagr > 0:  s += 2
    else:
        yoy = info.get('revenueGrowth')
        if yoy is not None:
            computed['revenue_growth_yoy_pct'] = round(yoy * 100, 2)
            r = yoy * 100
            if r > 20:   s += 6
            elif r > 10: s += 5
            elif r > 5:  s += 4
            elif r > 0:  s += 3
            elif r > -5: s += 1
        else:
            s += 4  # neutral (ETF / no data)

    # Earnings CAGR (0-9)
    earn_cagr = compute_earnings_cagr(financials)
    if earn_cagr is not None:
        computed['earnings_cagr_pct'] = earn_cagr
        if earn_cagr > 25:   s += 9
        elif earn_cagr > 20: s += 8
        elif earn_cagr > 15: s += 7
        elif earn_cagr > 10: s += 6
        elif earn_cagr > 5:  s += 4
        elif earn_cagr > 0:  s += 2
    else:
        yoy = info.get('earningsGrowth')
        if yoy is not None:
            computed['earnings_growth_yoy_pct'] = round(yoy * 100, 2)
            e = yoy * 100
            if e > 25:    s += 7
            elif e > 15:  s += 6
            elif e > 8:   s += 5
            elif e > 0:   s += 3
            elif e > -10: s += 1
        else:
            s += 4

    # FCF CAGR (0-8)
    fcf_cagr = compute_fcf_cagr(cashflow)
    if fcf_cagr is not None:
        computed['fcf_cagr_pct'] = fcf_cagr
        if fcf_cagr > 20:   s += 8
        elif fcf_cagr > 15: s += 7
        elif fcf_cagr > 10: s += 6
        elif fcf_cagr > 5:  s += 5
        elif fcf_cagr > 0:  s += 3
    else:
        fcf = info.get('freeCashflow')
        rev = info.get('totalRevenue')
        if fcf and rev and rev > 0:
            m = fcf / rev
            computed['fcf_margin_pct'] = round(m * 100, 2)
            if m > 0.20:   s += 6
            elif m > 0.12: s += 5
            elif m > 0.06: s += 4
            elif m > 0:    s += 3
        else:
            s += 4

    return min(s, 25), computed


def score_quality(info: dict) -> int:
    """ROE, debt/equity, operating margin → 0-20 pts."""
    s = 0

    # Return on equity (0-7)
    roe = info.get('returnOnEquity')
    if roe is not None:
        p = roe * 100
        if p > 25:   s += 7
        elif p > 20: s += 6
        elif p > 15: s += 5
        elif p > 10: s += 4
        elif p > 5:  s += 2
        elif p > 0:  s += 1
    else:
        s += 3

    # Debt/equity — yfinance reports as pct (e.g. 45 = 0.45×) (0-7)
    de = info.get('debtToEquity')
    if de is not None:
        if de < 30:    s += 7
        elif de < 60:  s += 6
        elif de < 100: s += 5
        elif de < 150: s += 3
        elif de < 200: s += 1
        # >200: 0
    else:
        s += 3

    # Operating margin (0-6)
    op = info.get('operatingMargins')
    if op is not None:
        p = op * 100
        if p > 30:   s += 6
        elif p > 20: s += 5
        elif p > 15: s += 4
        elif p > 10: s += 3
        elif p > 5:  s += 2
        elif p > 0:  s += 1
    else:
        s += 2

    return min(s, 20)


def score_valuation(info: dict) -> int:
    """P/E (absolute + vs growth), PEG, P/FCF → 0-20 pts."""
    s = 0
    pe = info.get('trailingPE')
    fpe = info.get('forwardPE')
    peg = info.get('pegRatio')
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')

    # P/E score (0-8) — lower absolute P/E is better value signal
    if pe and pe > 0:
        if pe < 12:    s += 8
        elif pe < 17:  s += 7
        elif pe < 22:  s += 6
        elif pe < 28:  s += 5
        elif pe < 35:  s += 3
        elif pe < 50:  s += 2
        elif pe < 80:  s += 1
        # >80: 0
    else:
        s += 4  # ETF / no earnings

    # PEG ratio (0-6)
    if peg and 0 < peg < 10:
        if peg < 0.7:   s += 6
        elif peg < 1.0: s += 5
        elif peg < 1.3: s += 4
        elif peg < 1.7: s += 3
        elif peg < 2.5: s += 2
    elif fpe and pe and pe > 0 and fpe > 0:
        improvement = (pe - fpe) / pe
        if improvement > 0.20:   s += 5
        elif improvement > 0.10: s += 4
        elif improvement > 0:    s += 3
        else:                    s += 2
    else:
        s += 3

    # P/FCF (0-6)
    if fcf and mcap and fcf > 0:
        p_fcf = mcap / fcf
        if p_fcf < 15:    s += 6
        elif p_fcf < 22:  s += 5
        elif p_fcf < 30:  s += 4
        elif p_fcf < 40:  s += 3
        elif p_fcf < 55:  s += 2
        # >55: 0
    else:
        s += 3

    return min(s, 20)


def score_stability(info: dict, dividends, cashflow) -> tuple[int, dict]:
    """Beta / volatility, dividend growth, earnings quality → 0-15 pts."""
    s = 0
    computed = {}

    # Beta (0-5) — lower = more stable
    beta = info.get('beta')
    if beta is not None:
        computed['beta'] = round(beta, 2)
        if beta < 0.5:   s += 5
        elif beta < 0.8: s += 4
        elif beta < 1.1: s += 3
        elif beta < 1.5: s += 2
        elif beta < 2.0: s += 1
    else:
        s += 2

    # Dividend growth CAGR (0-5); non-payers scored on ROA instead
    div_g = compute_dividend_growth(dividends)
    if div_g is not None:
        computed['dividend_growth_cagr_pct'] = div_g
        if div_g > 15:   s += 5
        elif div_g > 10: s += 4
        elif div_g > 5:  s += 3
        elif div_g > 0:  s += 2
        elif div_g > -5: s += 1
    else:
        roa = info.get('returnOnAssets')
        if roa and roa > 0.12:  s += 3
        elif roa and roa > 0.07: s += 2
        elif roa and roa > 0:    s += 1
        else:                    s += 1

    # Earnings quality: FCF / Net Income > 1 means cash-backed earnings (0-5)
    fcf = info.get('freeCashflow')
    ni = info.get('netIncomeToCommon')
    if fcf is not None and ni and ni > 0:
        ratio = fcf / ni
        computed['fcf_to_net_income'] = round(ratio, 2)
        if ratio > 1.3:   s += 5
        elif ratio > 1.0: s += 4
        elif ratio > 0.8: s += 3
        elif ratio > 0.5: s += 2
        elif ratio > 0:   s += 1
    else:
        s += 2

    return min(s, 15), computed

# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(hist, n_sims: int = 1000, horizon_days: int = 252) -> dict:
    """Geometric Brownian Motion simulation over horizon_days trading days."""
    close = hist['Close'].values.astype(float)
    if len(close) < 30:
        return {}

    log_returns = np.diff(np.log(close))
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns))
    current = float(close[-1])

    # Simulate: each row is one path, each column one day
    rand = np.random.standard_normal((n_sims, horizon_days))
    daily = np.exp((mu - 0.5 * sigma ** 2) + sigma * rand)
    final_prices = current * np.cumprod(daily, axis=1)[:, -1]

    p10  = float(np.percentile(final_prices, 10))
    p50  = float(np.percentile(final_prices, 50))
    p90  = float(np.percentile(final_prices, 90))
    prob = float(np.mean(final_prices > current) * 100)

    return {
        'mc_current_price':   round(current, 2),
        'mc_bear_price':      round(p10, 2),
        'mc_median_price':    round(p50, 2),
        'mc_bull_price':      round(p90, 2),
        'mc_prob_positive':   round(prob, 1),
        'mc_median_return':   round((p50 / current - 1) * 100, 1),
        'mc_simulations':     n_sims,
        'mc_horizon_days':    horizon_days,
    }

# ── Signal extraction ─────────────────────────────────────────────────────────

def extract_price_signals(hist) -> dict:
    close = hist['Close'].values.astype(float)
    volume = hist['Volume'].values.astype(float)
    signals = {'current_price': round(float(close[-1]), 4)}

    for days, key in [(2, 'ret_1d'), (6, 'ret_5d'), (21, 'ret_20d'),
                      (63, 'ret_3mo'), (126, 'ret_6mo'), (252, 'ret_1y')]:
        if len(close) >= days:
            signals[key] = round((close[-1] / close[-days] - 1) * 100, 2)

    rsi = compute_rsi(close)
    if rsi is not None:
        signals['rsi_14'] = rsi

    for n, label in [(20, 'ma_20'), (50, 'ma_50'), (200, 'ma_200')]:
        if len(close) >= n:
            ma = float(np.mean(close[-n:]))
            signals[label] = round(ma, 4)
            signals[f'price_vs_{label}_pct'] = round((close[-1] / ma - 1) * 100, 2)

    yr = close[-252:] if len(close) >= 252 else close
    signals['pct_from_52w_high'] = round((close[-1] / np.max(yr) - 1) * 100, 2)
    signals['pct_from_52w_low'] = round((close[-1] / np.min(yr) - 1) * 100, 2)

    if len(volume) >= 20 and np.mean(volume[-20:]) > 0:
        signals['volume_ratio'] = round(float(volume[-1] / np.mean(volume[-20:])), 2)

    if len(close) >= 30:
        daily_rets = np.diff(close[-63:]) / close[-63:-1]
        signals['volatility_ann_pct'] = round(float(np.std(daily_rets) * np.sqrt(252) * 100), 2)

    return signals


def extract_fundamentals(info: dict) -> dict:
    fields = [
        'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook',
        'priceToSalesTrailing12Months', 'profitMargins', 'grossMargins',
        'operatingMargins', 'debtToEquity', 'currentRatio',
        'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth',
        'dividendYield', 'dividendRate', 'freeCashflow', 'operatingCashflow',
        'totalRevenue', 'netIncomeToCommon', 'totalDebt', 'marketCap',
        'beta', 'sector', 'industry', 'shortName', 'longName',
    ]
    return {k: info[k] for k in fields if info.get(k) is not None}

# ── Claude narrative (scores already computed) ────────────────────────────────

def generate_narrative(asset: dict, scores: dict, price_signals: dict, fundamentals: dict) -> dict:
    client = anthropic.Anthropic()

    system = """You are an investment analyst. Given pre-computed quantitative scores and market data, write a concise narrative. Output ONLY valid JSON — no markdown.

Schema:
{
  "risk_level": "Low" | "Medium" | "High",
  "confidence": <integer 0-100, lower if data is sparse>,
  "time_horizon": "Short (< 3mo)" | "Medium (3-12mo)" | "Long (> 1yr)",
  "reasoning": {
    "situation": "<1-2 sentences with specific numbers>",
    "key_signal": "<the single most important data point>",
    "risk_factor": "<the main thing that could go wrong>",
    "why_now": "<specific reason to act now or wait>"
  }
}

Treat the quantitative scores as ground truth. Reference actual numbers from the data."""

    skip = {'shortName', 'longName', 'sector', 'industry',
            'ma_20', 'ma_50', 'ma_200', 'current_price'}
    key_metrics = {k: v for k, v in {**price_signals, **fundamentals}.items() if k not in skip}

    user = f"""Asset: {asset['ticker']} ({asset['market']})

Quantitative Scores:
  Momentum  {scores['momentum_score']}/20
  Growth    {scores['financial_score']}/25
  Quality   {scores['sentiment_score']}/20
  Valuation {scores['industry_score']}/20
  Stability {scores['valuation_score']}/15
  TOTAL     {scores['total_score']}/100

Key Metrics:
{json.dumps(key_metrics, indent=2)}"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)

# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_asset(asset: dict) -> Optional[dict]:
    yf_ticker = get_yf_ticker(asset['ticker'], asset['market'])
    try:
        t = yf.Ticker(yf_ticker)
        hist = t.history(period="1y")

        if hist.empty or len(hist) < 5:
            print(f"  Insufficient data for {yf_ticker}")
            return None

        info = t.info or {}

        try: financials = t.financials
        except Exception: financials = None
        try: cashflow = t.cashflow
        except Exception: cashflow = None
        try: dividends = t.dividends
        except Exception: dividends = None

        price_signals = extract_price_signals(hist)
        fundamentals = extract_fundamentals(info)
        name = fundamentals.pop('shortName', None) or fundamentals.pop('longName', None)

        momentum  = score_momentum(price_signals)
        growth,   growth_data   = score_growth(info, financials, cashflow)
        quality   = score_quality(info)
        valuation = score_valuation(info)
        stability, stab_data    = score_stability(info, dividends, cashflow)

        total = round(momentum + growth + quality + valuation + stability, 1)

        scores = {
            'momentum_score':  momentum,
            'financial_score': growth,     # stored as "financial", displayed as "Growth"
            'sentiment_score': quality,    # stored as "sentiment", displayed as "Quality"
            'industry_score':  valuation,  # stored as "industry",  displayed as "Valuation"
            'valuation_score': stability,  # stored as "valuation", displayed as "Stability"
            'total_score':     total,
        }

        narrative = generate_narrative(asset, scores, price_signals, fundamentals)
        mc = run_monte_carlo(hist)

        all_signals = {**price_signals, **growth_data, **stab_data, **fundamentals, **mc}

        return {
            'asset_id':        asset['id'],
            'name':            name,
            'price':           price_signals['current_price'],
            'price_change_pct': price_signals.get('ret_1d', 0),
            **scores,
            'risk_level':      narrative['risk_level'],
            'confidence':      narrative['confidence'],
            'time_horizon':    narrative['time_horizon'],
            'reasoning_json':  json.dumps(narrative['reasoning']),
            'signals_json':    json.dumps(all_signals),
        }

    except Exception as e:
        print(f"  Error analyzing {yf_ticker}: {e}")
        return None


def suggest_stocks(current_watchlist: list) -> list:
    client = anthropic.Anthropic()
    current = [f"{a['ticker']} ({a['market']})" for a in current_watchlist]

    system = """You are a stock discovery tool for Sharesies (NZ investing app).
Markets available on Sharesies: NZX (New Zealand), ASX (Australia), US/NYSE/NASDAQ (United States).
Output ONLY a valid JSON array — no markdown, no explanation.

Schema:
[{"ticker": "...", "market": "NZX"|"ASX"|"US"|"NYSE"|"NASDAQ", "company_name": "...", "why_interesting": "<2-3 specific sentences referencing industry trends, market position, or financials>", "theme": "growth"|"dividend"|"defensive"|"speculative"|"ETF", "industry": "<specific industry/sector, e.g. Aerospace & Defense, Dairy & Agriculture, Mining & Resources, Technology, Energy, Healthcare, Financial Services, Consumer, Real Estate, Utilities, Telecommunications, Industrials>"}]

Rules:
- Only use real tickers that actually trade on the stated exchange and are available on Sharesies
- Do not suggest tickers already in the watchlist
- Suggest exactly 5 stocks or ETFs
- Mix markets, themes, and industries for diversification"""

    user = f"""Current watchlist: {', '.join(current) if current else 'none yet'}

Suggest 5 stocks or ETFs available on Sharesies that would complement this watchlist."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def get_price_preview(ticker: str, market: str) -> dict:
    yf_ticker = get_yf_ticker(ticker, market)
    try:
        t = yf.Ticker(yf_ticker)
        hist = t.history(period="5d")
        if hist.empty or len(hist) < 2:
            return {}
        close = hist['Close'].values.astype(float)
        if np.isnan(close[-1]) or np.isnan(close[-2]):
            return {}
        info = t.info or {}
        return {
            'price': round(float(close[-1]), 4),
            'ret_1d': round((close[-1] / close[-2] - 1) * 100, 2),
            'name': info.get('shortName') or info.get('longName'),
            'sector': info.get('sector'),
        }
    except Exception:
        return {}
