import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel

load_dotenv()

from database import (
    init_db, add_asset, remove_asset, get_active_assets,
    save_snapshot, get_latest_snapshots, get_asset_history,
    update_asset_name, set_sharesies_flag,
    save_recommendations, get_latest_recommendations,
    create_user, authenticate_user, create_session,
    get_session_user, delete_session, get_username,
    is_admin, admin_exists, get_all_users, get_platform_stats,
    get_health_warnings, get_pending_users, approve_user, reject_user,
    verify_birthday, update_password,
)
from analyzer import analyze_asset, suggest_stocks, get_price_preview

scheduler = BackgroundScheduler()

VALID_MARKETS = {'US', 'NYSE', 'NASDAQ', 'ASX', 'NZX'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    interval_hours = int(os.getenv("REFRESH_HOURS", "6"))
    scheduler.add_job(refresh_all, 'interval', hours=interval_hours, id='refresh_all')
    scheduler.start()
    print(f"Investment Indicator started — auto-refresh every {interval_hours}h")
    yield
    scheduler.shutdown()

app = FastAPI(title="Investment Indicator", lifespan=lifespan)

class AddAssetRequest(BaseModel):
    ticker: str
    market: str
    sharesies_available: bool = True

class AuthRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = None
    birthday: str = None
    phone: str = None
    address: str = None

class VerifyBirthdayRequest(BaseModel):
    username: str
    birthday: str

class ResetPasswordRequest(BaseModel):
    username: str
    new_password: str

def require_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization[7:]
    user_id = get_session_user(token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired session")
    return user_id

def require_admin(user_id: str = Depends(require_user)) -> str:
    if not is_admin(user_id):
        raise HTTPException(403, "Admin access required")
    return user_id

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    username = req.username.strip()
    if len(username) < 3 or len(username) > 50:
        raise HTTPException(400, "Username must be 3–50 characters")
    if not username.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(400, "Username may only contain letters, numbers, hyphens and underscores")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        create_user(username, req.password,
                    email=req.email, birthday=req.birthday,
                    phone=req.phone, address=req.address)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"pending": True, "username": username.lower()}

@app.post("/api/auth/login")
def login(req: AuthRequest):
    user_id, status = authenticate_user(req.username, req.password)
    if not user_id:
        raise HTTPException(401, "Invalid username or password")
    if status == 'pending':
        raise HTTPException(403, "Your account is awaiting admin approval")
    if status == 'rejected':
        raise HTTPException(403, "Your account request was not approved")
    token = create_session(user_id)
    return {"token": token, "username": req.username.lower().strip()}

@app.post("/api/auth/verify-birthday")
def verify_birthday_endpoint(req: VerifyBirthdayRequest):
    if not verify_birthday(req.username, req.birthday):
        raise HTTPException(400, "Username or date of birth is incorrect")
    return {"ok": True}

@app.post("/api/auth/reset-password")
def reset_password_endpoint(req: ResetPasswordRequest):
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    update_password(req.username, req.new_password)
    return {"ok": True}

@app.post("/api/auth/logout")
def logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        delete_session(authorization[7:])
    return {"ok": True}

@app.get("/api/auth/me")
def me(user_id: str = Depends(require_user)):
    return {"user_id": user_id, "username": get_username(user_id), "is_admin": is_admin(user_id)}

@app.get("/api/admin/setup-status")
def admin_setup_status():
    return {"setup_needed": not admin_exists()}

@app.post("/api/admin/setup")
def admin_setup(req: AuthRequest):
    if admin_exists():
        raise HTTPException(409, "An admin account already exists")
    username = req.username.strip()
    if len(username) < 3 or len(username) > 50:
        raise HTTPException(400, "Username must be 3–50 characters")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        user_id = create_user(username, req.password, is_admin=True)
    except ValueError as e:
        raise HTTPException(409, str(e))
    token = create_session(user_id)
    return {"token": token, "username": username.lower()}

@app.get("/api/admin/stats")
def admin_stats(_: str = Depends(require_admin)):
    return get_platform_stats()

@app.get("/api/admin/users")
def admin_users(_: str = Depends(require_admin)):
    return get_all_users()

@app.get("/api/admin/health")
def admin_health(_: str = Depends(require_admin)):
    return get_health_warnings()

@app.get("/api/admin/requests")
def admin_requests(_: str = Depends(require_admin)):
    return get_pending_users()

@app.post("/api/admin/users/{user_id}/approve")
def admin_approve(user_id: str, _: str = Depends(require_admin)):
    approve_user(user_id)
    return {"ok": True}

@app.post("/api/admin/users/{user_id}/reject")
def admin_reject(user_id: str, _: str = Depends(require_admin)):
    reject_user(user_id)
    return {"ok": True}

@app.get("/api/admin/users/{user_id}/assets")
def admin_user_assets(user_id: str, _: str = Depends(require_admin)):
    return get_active_assets(user_id)

@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: str, admin_id: str = Depends(require_admin)):
    if user_id == admin_id:
        raise HTTPException(400, "Cannot delete your own account")
    from database import get_db
    from sqlalchemy import text as _text
    with get_db() as conn:
        conn.execute(_text("DELETE FROM sessions WHERE user_id = :uid"), {'uid': user_id})
        conn.execute(_text("UPDATE assets SET active = 0 WHERE user_id = :uid"), {'uid': user_id})
        conn.execute(_text("DELETE FROM users WHERE user_id = :uid"), {'uid': user_id})
    return {"ok": True}

def refresh_all():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Auto-refreshing all users...")
    assets = get_active_assets()  # all users
    for asset in assets:
        result = analyze_asset(asset)
        if result:
            if result.get('name'):
                update_asset_name(asset['id'], result['name'])
            save_snapshot(result)
            print(f"  ✓ {asset['ticker']}: {result['total_score']}/100")
        else:
            print(f"  ✗ {asset['ticker']}: skipped (no data)")

def analyze_in_background(asset: dict):
    result = analyze_asset(asset)
    if result:
        if result.get('name'):
            update_asset_name(asset['id'], result['name'])
        save_snapshot(result)

@app.get("/api/watchlist")
def get_watchlist(user_id: str = Depends(require_user)):
    return get_active_assets(user_id)

@app.post("/api/watchlist")
def post_add_asset(req: AddAssetRequest, user_id: str = Depends(require_user)):
    ticker = req.ticker.strip().upper()
    market = req.market.strip().upper()
    if not ticker:
        raise HTTPException(400, "Ticker is required")
    if market not in VALID_MARKETS:
        raise HTTPException(400, f"Market must be one of: {', '.join(sorted(VALID_MARKETS))}")
    asset = add_asset(ticker, market, req.sharesies_available, user_id)
    threading.Thread(target=analyze_in_background, args=(asset,), daemon=True).start()
    return asset

@app.delete("/api/watchlist/{asset_id}")
def delete_asset(asset_id: int):
    remove_asset(asset_id)
    return {"ok": True}

@app.patch("/api/watchlist/{asset_id}/sharesies")
def patch_sharesies(asset_id: int, available: bool):
    set_sharesies_flag(asset_id, available)
    return {"ok": True}

@app.get("/api/snapshots")
def get_snapshots(user_id: str = Depends(require_user)):
    rows = get_latest_snapshots(user_id)
    for row in rows:
        if row.get('reasoning_json'):
            row['reasoning'] = json.loads(row['reasoning_json'])
        if row.get('signals_json'):
            row['signals'] = json.loads(row['signals_json'])
    return rows

@app.get("/api/history/{asset_id}")
def get_history(asset_id: int):
    rows = get_asset_history(asset_id)
    for row in rows:
        if row.get('reasoning_json'):
            row['reasoning'] = json.loads(row['reasoning_json'])
    return rows

@app.post("/api/refresh")
def manual_refresh_all(user_id: str = Depends(require_user)):
    assets = get_active_assets(user_id)
    def _run():
        for asset in assets:
            result = analyze_asset(asset)
            if result:
                if result.get('name'):
                    update_asset_name(asset['id'], result['name'])
                save_snapshot(result)
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}

@app.post("/api/refresh/{asset_id}")
def manual_refresh_one(asset_id: int, user_id: str = Depends(require_user)):
    assets = get_active_assets(user_id)
    asset = next((a for a in assets if a['id'] == asset_id), None)
    if not asset:
        raise HTTPException(404, "Asset not found")
    threading.Thread(target=analyze_in_background, args=(asset,), daemon=True).start()
    return {"ok": True}

@app.post("/api/suggest")
def get_suggestions(user_id: str = Depends(require_user)):
    current_assets = get_active_assets(user_id)
    current_set = {(a['ticker'], a['market']) for a in current_assets}
    try:
        suggestions = suggest_stocks(current_assets)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate suggestions: {e}")

    candidates = []
    for s in suggestions:
        ticker = s['ticker'].upper()
        market = s['market'].upper()
        if (ticker, market) in current_set:
            continue
        candidates.append((ticker, market, s))

    previews = [None] * len(candidates)
    def _fetch(i, ticker, market):
        previews[i] = get_price_preview(ticker, market)
    threads = [
        threading.Thread(target=_fetch, args=(i, ticker, market))
        for i, (ticker, market, _) in enumerate(candidates)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    results = []
    for (ticker, market, s), preview in zip(candidates, previews):
        results.append({
            'ticker': ticker,
            'market': market,
            'company_name': s.get('company_name', ''),
            'why_interesting': s.get('why_interesting', ''),
            'theme': s.get('theme', ''),
            'price': preview.get('price'),
            'ret_1d': preview.get('ret_1d'),
            'name': preview.get('name') or s.get('company_name', ''),
            'sector': preview.get('sector'),
            'valid': bool(preview),
        })
    return results

@app.get("/api/recommendations")
def list_recommendations(user_id: str = Depends(require_user)):
    recs = get_latest_recommendations(user_id)
    previews = [None] * len(recs)
    def _fetch(i, ticker, market):
        previews[i] = get_price_preview(ticker, market)
    threads = [
        threading.Thread(target=_fetch, args=(i, r['ticker'], r['market']))
        for i, r in enumerate(recs)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    results = []
    for r, preview in zip(recs, previews):
        current_price = preview.get('price')
        change_pct = None
        if current_price is not None and r['price_at_rec']:
            change_pct = round((current_price / r['price_at_rec'] - 1) * 100, 2)
        results.append({**r, 'current_price': current_price, 'change_pct': change_pct})
    return results

@app.post("/api/recommendations/refresh")
def refresh_recommendations(user_id: str = Depends(require_user)):
    current_assets = get_active_assets(user_id)
    current_set = {(a['ticker'], a['market']) for a in current_assets}
    try:
        suggestions = suggest_stocks(current_assets)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate recommendations: {e}")

    candidates = [
        s for s in suggestions
        if (s['ticker'].upper(), s['market'].upper()) not in current_set
    ]
    previews = [None] * len(candidates)
    def _fetch(i, ticker, market):
        previews[i] = get_price_preview(ticker, market)
    threads = [
        threading.Thread(target=_fetch, args=(i, s['ticker'].upper(), s['market'].upper()))
        for i, s in enumerate(candidates)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    items = []
    for s, preview in zip(candidates, previews):
        items.append({
            'ticker': s['ticker'].upper(),
            'market': s['market'].upper(),
            'company_name': s.get('company_name', ''),
            'why_interesting': s.get('why_interesting', ''),
            'theme': s.get('theme', ''),
            'industry': s.get('industry', ''),
            'price_at_rec': preview.get('price'),
        })
    save_recommendations(user_id, str(uuid.uuid4()), items)
    return {"ok": True, "count": len(items)}

@app.get("/api/prices/{asset_id}")
def get_prices(asset_id: int, period: str = "1y"):
    from analyzer import get_yf_ticker
    import yfinance as yf
    if period not in {"1y", "5y"}:
        raise HTTPException(400, "period must be 1y or 5y")
    assets = get_active_assets()
    asset = next((a for a in assets if a['id'] == asset_id), None)
    if not asset:
        raise HTTPException(404, "Asset not found")
    try:
        t = yf.Ticker(get_yf_ticker(asset['ticker'], asset['market']))
        hist = t.history(period=period, interval="1wk" if period == "5y" else "1d")
        if hist.empty:
            return []
        return [
            {"date": str(idx.date()), "price": round(float(row['Close']), 4)}
            for idx, row in hist.iterrows()
        ]
    except Exception:
        return []

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
