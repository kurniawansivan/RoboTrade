"""
Bot entry point. Wires all layers together and starts the WebSocket loop.
Execution flow per candle:
  data/ingestion → features → strategy.generate_signal
  → risk.approve_signal → execution.place_bracket_order
  → portfolio.sync_positions (background every 5s)
"""

import asyncio
import logging
import os
import signal
import sys

import redis.asyncio as aioredis
import yaml
from dotenv import load_dotenv

import ccxt.pro as ccxtpro
from data.ingestion import run_ws_loop
from data.storage import build_engine, init_schema
from execution.order_manager import (
    cancel_all_orders,
    close_all_positions,
    close_position,
    place_bracket_order,
)
from execution.portfolio import (
    get_all_positions,
    get_balance,
    get_day_start_balance,
    has_open_position,
    run_position_monitor,
    sync_balance,
    init_day_start_balance,
)
from monitoring.telegram_alerts import (
    alert_bot_started,
    alert_bot_stopped,
    alert_trade_opened,
    alert_drawdown_gate,
    alert_daily_summary,
    alert_error,
)
from risk.manager import RiskManager
from strategy.rule_based import RuleBasedStrategy

load_dotenv(dotenv_path="config/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # LEVERAGE env var overrides config (no hard cap — see config risk.max_leverage guard)
    lev_env = os.environ.get("LEVERAGE")
    if lev_env:
        try:
            cfg["risk"]["leverage"] = int(lev_env)
        except ValueError:
            logger.warning("invalid LEVERAGE env value: %s", lev_env)
    return cfg


def _make_strategy(strategy_type: str):
    """Factory: pick strategy implementation by config type."""
    if strategy_type == "ml":
        from strategy.ml_strategy import MLStrategy
        return MLStrategy()
    return RuleBasedStrategy()


def _build_exchange(config: dict) -> ccxtpro.Exchange:
    exchange_cls = getattr(ccxtpro, config["exchange"]["name"])
    exchange = exchange_cls(
        {
            "apiKey": os.environ["BITGET_API_KEY"],
            "secret": os.environ["BITGET_API_SECRET"],
            "password": os.environ["BITGET_API_PASSPHRASE"],
        }
    )
    exchange.set_sandbox_mode(config["exchange"].get("sandbox", True))
    return exchange


async def main() -> None:
    config = load_config()
    sandbox: bool = config["exchange"].get("sandbox", True)

    if not sandbox:
        logger.warning("⚠️  LIVE MODE — real funds at risk. Confirm config.yaml: sandbox: false")

    # ── Infrastructure ─────────────────────────────────────────────────────
    engine = build_engine(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        db=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    init_schema(engine)
    logger.info("DB schema ready")

    redis_client = aioredis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        decode_responses=True,
    )
    await redis_client.ping()
    logger.info("Redis connected")

    exchange = _build_exchange(config)
    symbols: list[str] = config["exchange"].get("symbols", [config["exchange"]["symbol"]])
    max_positions: int = int(config["risk"].get("max_open_positions", 3))

    # ── Initial balance sync ───────────────────────────────────────────────
    await sync_balance(exchange, redis_client, symbols[0])
    await init_day_start_balance(redis_client)
    balance = await get_balance(redis_client)
    logger.info("initial balance", extra={"balance_usdt": balance, "symbols": symbols, "sandbox": sandbox})

    # ── Strategy + Risk ────────────────────────────────────────────────────
    strategy = _make_strategy(config["strategy"].get("type", "rule_based"))
    risk_manager = RiskManager()

    # Fetch per-symbol exchange limits + set leverage + force one-way position mode
    await exchange.load_markets()
    leverage = int(config["risk"]["leverage"])
    if leverage > 20:
        logger.warning("⚠️  HIGH LEVERAGE: %d× — large losses possible. Set via LEVERAGE env.", leverage)

    # Force one-way (unilateral) position mode account-wide — avoids hedge-mode
    # order-type mismatch (Bitget error 40774).
    try:
        await exchange.set_position_mode(False, symbols[0], params={"productType": "USDT-FUTURES"})
        logger.info("position mode set to one-way")
    except Exception as exc:
        logger.warning("set_position_mode failed (may already be one-way): %s", exc)

    sym_limits: dict[str, dict] = {}
    for sym in symbols:
        try:
            m = exchange.market(sym)
            sym_limits[sym] = {
                "min_qty": float(m.get("limits", {}).get("amount", {}).get("min") or 0.001),
                "min_notional": float(m.get("limits", {}).get("cost", {}).get("min") or 5.0),
            }
        except Exception:
            sym_limits[sym] = {"min_qty": 0.001, "min_notional": 5.0}
        # Set leverage on the exchange for this symbol (best-effort)
        try:
            await exchange.set_leverage(leverage, sym, params={"productType": "USDT-FUTURES"})
        except Exception as exc:
            logger.warning("set_leverage failed for %s: %s", sym, exc)
    logger.info("exchange ready | symbols=%s leverage=%d×", list(sym_limits.keys()), leverage)

    # ── Candle callback (called per symbol) ───────────────────────────────
    import json as _json

    async def on_candle(sym: str, df_features) -> None:
        """Called by each symbol's ingestion loop on every closed candle."""
        if len(df_features) < 2:
            return

        cfg = load_config()
        last = df_features.iloc[-1]
        price = float(last["close"])

        # Write per-symbol indicator state to Redis
        signal_now = strategy.generate_signal(df_features, cfg["strategy"], symbol=sym)
        indicator_state = {
            "symbol": sym,
            "signal": signal_now or "none",
            "price": round(price, 2),
            "rsi": round(float(last.get("rsi", 0)), 2),
            "ema_fast": round(float(last.get("ema_fast", 0)), 2),
            "ema_slow": round(float(last.get("ema_slow", 0)), 2),
            "macd_hist": round(float(last.get("macd_hist", 0)), 4),
            "atr": round(float(last.get("atr", 0)), 2),
            "htf_trend": int(last.get("htf_trend", 0)),
            "bar_ts": str(last.get("open_time", "")),
        }
        redis_key = f"indicator_state:{sym.replace('/', '_')}"
        await redis_client.set(redis_key, _json.dumps(indicator_state))
        # Also write "latest" key so dashboard always shows most recent
        await redis_client.set("indicator_state", _json.dumps(indicator_state))

        htf_label = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(indicator_state["htf_trend"], "?")
        logger.info(
            "%-18s price=%-12s RSI=%-6s MACD_h=%-10s 1h=%s | signal=%s",
            sym,
            f"${price:,.2f}",
            f"{indicator_state['rsi']:.1f}",
            f"{indicator_state['macd_hist']:.4f}",
            htf_label,
            (signal_now or "none").upper(),
        )

        if signal_now is None:
            return

        # ── Halt flag (set from dashboard) blocks new entries ──────────────
        if await redis_client.get("bot:halted") == "1":
            logger.info("trading halted — skipping %s entry", sym)
            return

        # ── Total position cap across all symbols ──────────────────────────
        all_positions = await get_all_positions(redis_client, symbols)
        if len(all_positions) >= max_positions:
            logger.info("max positions reached (%d), skipping %s", max_positions, sym)
            return

        # Skip if this specific symbol already has a position
        if await has_open_position(redis_client, sym):
            return

        atr = float(last["atr"])
        balance = await get_balance(redis_client)
        day_start = await get_day_start_balance(redis_client)
        lims = sym_limits.get(sym, {"min_qty": 0.001, "min_notional": 5.0})

        spec = risk_manager.approve_signal(
            signal=signal_now,
            price=price,
            atr=atr,
            balance=balance,
            day_start_balance=day_start,
            config=cfg,
            min_qty=lims["min_qty"],
            min_notional=lims["min_notional"],
        )
        if spec is None:
            if day_start > 0:
                dd = (day_start - balance) / day_start
                if dd >= cfg["risk"]["max_daily_drawdown"]:
                    await alert_drawdown_gate(balance, day_start, dd * 100)
            return

        logger.info(
            "placing order on %s | side=%s qty=%s entry=%s sl=%s tp=%s",
            sym, spec.side, spec.qty, spec.entry, spec.sl, spec.tp,
        )
        order = await place_bracket_order(
            exchange=exchange,
            symbol=sym,
            spec=spec,
            engine=engine,
            strategy_type=cfg["strategy"]["type"],
        )
        if order:
            # Store intended protection levels for the software-side stop safety net
            await redis_client.set(
                f"protect:{sym.replace('/', '_')}",
                _json.dumps({"side": spec.side, "sl": spec.sl, "tp": spec.tp, "entry": spec.entry}),
            )
            await alert_trade_opened(
                signal=signal_now,
                side=spec.side,
                entry=spec.entry,
                sl=spec.sl,
                tp=spec.tp,
                qty=spec.qty,
                risk_usd=spec.risk_usd,
                symbol=sym,
            )

    # ── Multi-symbol position monitor (syncs balance + every symbol's position) ──
    from execution.portfolio import sync_positions

    async def _enforce_soft_stop(s: str, positions: list) -> None:
        """Software-side SL/TP backup: close if mark price breaches intended levels."""
        protect_key = f"protect:{s.replace('/', '_')}"
        raw = await redis_client.get(protect_key)
        open_now = [p for p in positions if float(p.get("contracts") or 0) != 0]
        if not open_now:
            # position gone (closed by exchange bracket or manually) → clear protect
            if raw:
                await redis_client.delete(protect_key)
            return
        if not raw:
            return
        prot = _json.loads(raw)
        pos = open_now[0]
        mark = float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0)
        if mark <= 0:
            return
        side, sl, tp = prot["side"], float(prot["sl"]), float(prot["tp"])
        hit = None
        if side == "buy":
            if mark <= sl: hit = "SL"
            elif mark >= tp: hit = "TP"
        else:
            if mark >= sl: hit = "SL"
            elif mark <= tp: hit = "TP"
        if hit:
            logger.warning("soft-stop %s hit on %s @ %.4f — closing", hit, s, mark)
            await close_position(exchange, s)
            await redis_client.delete(protect_key)

    async def multi_monitor() -> None:
        logger.info("position monitor started (all symbols)")
        while True:
            try:
                await sync_balance(exchange, redis_client, symbols[0])
                await init_day_start_balance(redis_client)
                for s in symbols:
                    positions = await sync_positions(exchange, redis_client, s)
                    await _enforce_soft_stop(s, positions)
            except Exception as exc:
                logger.error("monitor error", exc_info=exc)
            await asyncio.sleep(5.0)

    # ── Command consumer — executes dashboard commands from Redis ──────────
    async def command_consumer() -> None:
        logger.info("command consumer started")
        while True:
            try:
                raw = await redis_client.rpop("bot:commands")
                if raw:
                    cmd = _json.loads(raw)
                    action = cmd.get("action")
                    sym = cmd.get("symbol")
                    logger.info("command received: %s %s", action, sym or "")
                    if action == "close" and sym:
                        await close_position(exchange, sym)
                    elif action == "flatten_all":
                        for s in symbols:
                            await close_position(exchange, s)
                    elif action == "halt":
                        await redis_client.set("bot:halted", "1")
                        logger.warning("TRADING HALTED via dashboard")
                    elif action == "resume":
                        await redis_client.set("bot:halted", "0")
                        logger.info("trading resumed via dashboard")
                    await redis_client.set("bot:last_command", raw)
                else:
                    await asyncio.sleep(1.0)
            except Exception as exc:
                logger.error("command consumer error", exc_info=exc)
                await asyncio.sleep(1.0)

    # ── Signal handlers ────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("shutdown signal received", extra={"signal": sig.name})
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig)

    # ── Start tasks ────────────────────────────────────────────────────────
    await redis_client.set("bot:halted", "0")  # reset halt on startup
    monitor_task = asyncio.create_task(multi_monitor(), name="position_monitor")
    command_task = asyncio.create_task(command_consumer(), name="command_consumer")
    ws_tasks = [
        asyncio.create_task(
            run_ws_loop(
                engine=engine,
                redis_client=redis_client,
                config=config,
                strategy_config=config["strategy"],
                on_candle=on_candle,
                symbol_override=sym,
            ),
            name=f"ws_{sym}",
        )
        for sym in symbols
    ]

    logger.info(
        "bot started | symbols=%s strategy=%s sandbox=%s leverage=%s×",
        symbols, config["strategy"]["type"], sandbox, config["risk"]["leverage"],
    )
    await alert_bot_started(f"{len(symbols)} symbols: {', '.join(s.split('/')[0] for s in symbols)}", sandbox)

    await stop_event.wait()

    # ── Graceful shutdown ──────────────────────────────────────────────────
    logger.info("shutting down…")
    for t in ws_tasks:
        t.cancel()
    monitor_task.cancel()
    command_task.cancel()

    for t in ws_tasks + [monitor_task, command_task]:
        try:
            await t
        except asyncio.CancelledError:
            pass

    # Kill switch: cancel all open orders + close all positions across all symbols
    logger.info("kill switch: closing all positions and orders")
    for sym in symbols:
        await cancel_all_orders(exchange, sym)
        await close_all_positions(exchange, sym)

    # Daily summary + stopped alert
    final_balance = await get_balance(redis_client)
    day_start_final = await get_day_start_balance(redis_client)
    await alert_daily_summary(final_balance, day_start_final, n_trades=0, sandbox=sandbox)
    await alert_bot_stopped("manual stop")

    await exchange.close()
    await redis_client.aclose()
    engine.dispose()
    logger.info("bot stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
