"""cbBTC minute-bar trading bot for the make.fun bars-replay/v1 harness.

Strategy (long only, one position, paper money): capital preservation first.
The bot trades only two high-conviction setups on the bar closes the harness
sends in each batch, and stays in cash the rest of the time:

  1. Capitulation bounce: the close is at least 1.5% below the highest close
     of the last 360 bars (a deep drawdown) and short-term momentum has
     turned up (the close is at least 0.2% above its close 15 bars ago).
  2. Trend momentum: the close is at least 0.6% above its close 60 bars
     ago, a one-hour move large enough to clear the round-trip cost
     (about 26 bps at full size: 4 bps fee + 2 bps slippage + pool impact
     on each side).

Exits are mechanical: a hard stop-loss 0.7% under the position's cost, a
take-profit at +1.5% over cost, a trailing stop (armed at +0.8%, 0.7% under
the peak value), a momentum-fade exit when the 60-bar return falls to
-0.2% or worse, and a 600-bar maximum hold. After any exit the bot waits
30 bars before entering again, so chop cannot churn it. Every order caps
its slippage at 200 bps from the latest close.

Deterministic: pure arithmetic on the bar values in each batch; no clock,
no environment, no randomness, no network, standard library only. The
vendored SDK shim (vendor/makefun_bars.py, a byte-for-byte copy of the
kit's sdk/makefun_bars.py) owns the hello / batch / reply wire protocol:
it folds every batch into state, echoes the batch seq, and checks each
order before it is sent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"

# Entry parameters (fractions of price, bar counts).
DIP_WINDOW = 360          # bars of history behind the drawdown measure
DIP_DRAWDOWN = 0.015      # close at least this far under the window high
TURN_BARS = 15            # short momentum lookback for the bounce turn
TURN_UP = 0.002           # ...and at least this far up over it
MOM_BARS = 60             # one-hour momentum lookback
MOM_UP = 0.006            # one-hour return needed for a momentum entry

# Exit parameters (fractions of the position's cost, fees included).
STOP_LOSS = 0.007
TAKE_PROFIT = 0.015
TRAIL_ARM = 0.008
TRAIL_GIVEBACK = 0.007
FADE = -0.002             # 60-bar return at or under this exits a position
MAX_HOLD_BARS = 600
COOLDOWN_BARS = 30

MAX_SLIPPAGE_BPS = 200

# Every CBBTC close seen so far, oldest first (raw integer prices).
closes = []
# Position bookkeeping: the seq the current position was first seen on,
# and the best value/cost ratio since (for the trailing stop).
entry_seq = None
peak_ratio = 0.0
# The seq of the batch on which the last exit was decided.
last_exit_seq = -10 ** 9
# The seq of the batch on which a buy was last sent, while its fill has
# not appeared in the account yet; cleared one batch later if the market
# rejected the order, so a rejected buy never wedges the bot.
buy_sent_seq = None


def _return_over(bars_back):
    """Close now against the close bars_back bars ago, or None."""
    if len(closes) <= bars_back:
        return None
    then = closes[-1 - bars_back]
    if then == 0:
        return None
    return closes[-1] / then - 1.0


def on_batch(state, batch):
    global entry_seq, peak_ratio, last_exit_seq, buy_sent_seq

    # Fold this batch's bar into the history; a batch with no CBBTC bar
    # (a gap minute) adds nothing and is never decided on.
    fresh = False
    for bar in batch["bars"]:
        if bar["symbol"] == SYMBOL:
            closes.append(int(bar["c"]))
            fresh = True
    if not fresh or state.warming_up:
        return []
    close = state.close(SYMBOL)
    if close is None or close == 0:
        return []

    held = state.positions.get(SYMBOL)

    # --- Position management: one exit reason is enough. ---
    if held is not None:
        buy_sent_seq = None
        if entry_seq is None:
            entry_seq = state.seq
            peak_ratio = 0.0
        value = state.value(SYMBOL, held.qty)
        ratio = value / held.cost if held.cost > 0 else 0.0
        if ratio > peak_ratio:
            peak_ratio = ratio
        gain = ratio - 1.0
        fade = _return_over(MOM_BARS)
        exit_now = (
            gain <= -STOP_LOSS
            or gain >= TAKE_PROFIT
            or (peak_ratio - 1.0 >= TRAIL_ARM and gain <= peak_ratio - 1.0 - TRAIL_GIVEBACK)
            or (fade is not None and fade <= FADE)
            or state.seq - entry_seq >= MAX_HOLD_BARS
        )
        if exit_now:
            entry_seq = None
            peak_ratio = 0.0
            last_exit_seq = state.seq
            return [{"side": "sell", "symbol": SYMBOL, "qty": held.qty,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS}]
        return []

    # Flat. A buy sent on the previous batch has not filled yet: do not
    # stack orders. One batch on, a still-flat account means the market
    # rejected it, so the way is clear to decide again.
    entry_seq = None
    peak_ratio = 0.0
    if buy_sent_seq is not None:
        if state.seq - buy_sent_seq < 2:
            return []
        buy_sent_seq = None
    if state.seq - last_exit_seq < COOLDOWN_BARS:
        return []

    # --- Entries: either setup, sized to all the cash the caps allow. ---
    enter = False
    if len(closes) > DIP_WINDOW:
        high = max(closes[-DIP_WINDOW - 1:])
        turn = _return_over(TURN_BARS)
        if high > 0 and closes[-1] / high - 1.0 <= -DIP_DRAWDOWN \
                and turn is not None and turn >= TURN_UP:
            enter = True
    if not enter:
        mom = _return_over(MOM_BARS)
        if mom is not None and mom >= MOM_UP:
            enter = True
    if not enter:
        return []

    notional = min(state.cash, state.weight_room(SYMBOL))
    if notional < int(state.hello["fees"]["min_notional_quote_units"]):
        return []
    buy_sent_seq = state.seq
    return [{"side": "buy", "symbol": SYMBOL,
             "notional_quote_units": notional,
             "max_slippage_bps": MAX_SLIPPAGE_BPS}]


if __name__ == "__main__":
    makefun_bars.run(on_batch)
