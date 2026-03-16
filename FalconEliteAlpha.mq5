
Freedium
ko-fi
librepay
patreon
< Go to the original

Preview image
I Turned a Chinese Quant Finance Paper into a Forex Trading Bot — Here's Everything I Learned
From a 2020 academic paper about Chinese equities to a working MetaTrader 5 Expert Advisor with CVaR risk management — including every…
Javier Santiago Gastón de Iriarte Cabrera
Javier Santiago Gastón de Iriarte Cabrera

Follow
a11y-light
·
February 25, 2026 (Updated: February 25, 2026)
·
Free: No
From a 2020 academic paper about Chinese equities to a working MetaTrader 5 Expert Advisor with CVaR risk management — including every backtest that failed before one finally didn't
Most algorithmic trading articles start with a strategy that already works. This one doesn't.

This is the honest story of taking a peer-reviewed paper written for China's A-share equity market, stripping it to its mathematical core, and rebuilding those ideas as a working MetaTrader 5 Expert Advisor for forex. It includes the backtest that lost money, the three bugs that silently killed trades without a single error in the journal, and the decisions that eventually produced a Sharpe ratio of 2.79 on a full year of data.

By the end you'll find the complete source code and the real results. But more importantly, you'll understand why each piece of the system exists — not just what it does.

The Paper
"Financial Trading Strategy System Based on Machine Learning" — Chen et al., Mathematical Problems in Engineering, 2020

It's not a forex paper. It's not even a short-term trading paper. Chen et al. built a quarterly rebalancing system for Chinese A-share stocks using fundamental factors: ROE, P/E ratios, cash flow metrics, beta, turnover rate. Their universe was 50 companies over ten years (2008–2018).

So why does any of this matter for a currency trader?

Because the mathematical architecture underneath the stock-picking is completely universal. Three ideas in this paper apply to any tradeable instrument regardless of asset class or timeframe: multi-factor scoring with learned importance weights, CVaR-constrained position sizing, and minimum variance allocation. Strip away the Chinese equities context and you have a rigorous framework for combining weak signals into disciplined, risk-aware trading decisions. That's what we built.

Part 1 — What the Paper Actually Does
The three-layer architecture
The system runs three sequential stages. Every decision flows through this pipeline:

Copy
┌─────────────────────────────────────────┐
│         LAYER 1 — FACTOR SCORING        │
│  Input:  Financial / price factors      │
│  Model:  LightGBM gradient boosting     │
│  Output: Predicted return score         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       LAYER 2 — ASSET SELECTION         │
│  Rank by predicted score                │
│  Filter below minimum threshold         │
│  Select top candidates only             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     LAYER 3 — PORTFOLIO ALLOCATION      │
│  Minimize variance + CVaR constraint    │
│  Output: Position size per asset        │
└─────────────────────────────────────────┘
The model at the heart of Layer 1 is LightGBM. The paper compared three approaches:

Copy
Model        R² Score    Training Speed
────────────────────────────────────────
GLM           0.443          Fast
XGBoost       0.771          Slow
LightGBM      0.798       Fastest  ✓
LightGBM wins for two technical reasons. It grows trees leaf-wise rather than level-wise, finding better splits per iteration. And its Exclusive Feature Bundling compresses sparse correlated features without losing information — critical when you have dozens of financial ratios.

The variable importance chart — the real insight
Figure 4 of the paper is the most important thing in it. After training, LightGBM assigns an importance score to each factor:

Copy
Close_Price    ████████████████████████  235
AP             ████████                   48
Price_Rate     ██████                     42
Turnover_Rate  █████                      38
Beta           ████                       32
ROE            ███                        28
P/E Ratio      ██                         19
Close_Price scores 235. The second factor scores 48. Price is nearly five times more predictive than any fundamental metric. The model, trained purely on data, revealed that this is a momentum system with a fundamental overlay — not the other way around. This single finding completely determines how we adapt the paper to forex.

Why CVaR and not just a stop loss
The paper uses CVaR — Conditional Value at Risk — as its risk constraint. Most traders know VaR, but CVaR is fundamentally more useful for fat-tailed distributions like forex returns.

Copy
Return distribution:
  Frequency
     |      ╭────╮
     |    ╭─╯    ╰─╮
     |  ╭─╯        ╰─╮
     |╭─╯             ╰──────────────
     └────────────────────────────── Return
                   │         │
                  VaR       CVaR
                 (5%)       (5%)
                   │         │
             "You lose     "On your worst
             less than     days, you lose
             this 95%      THIS on average"
             of days"
VaR only tells you the threshold. CVaR tells you what happens beyond it — the expected loss in the worst 5% of scenarios. For fat-tailed distributions, CVaR can be dramatically larger than VaR. If you're only measuring VaR, you're blind to the risk that actually matters most.

The paper's formula for parametric CVaR:

Copy
CVaR = -(μ - σ × φ(z) / (1 - α))
  μ      = mean log return over lookback window
  σ      = standard deviation of returns
  φ(z)   = standard normal PDF at confidence level z
  α      = confidence level (0.95)
Part 2 — The Translation Problem
You cannot run LightGBM inside MetaTrader 5. There is no Python runtime, no model serialization, no way to load a trained gradient boosting model into MQL5. This is the core engineering challenge.

The solution is to replicate the output behavior of LightGBM rather than the model itself. A trained model takes multiple input features and returns a score between 0 and 1. We can build a weighted scoring system that does exactly the same thing — using the feature importance values from the paper directly as weights.

Replacing fundamental factors with forex equivalents
The paper's factors are quarterly balance sheet data. Forex has no balance sheet. We need technical signals that capture the same underlying phenomena:

Copy
PAPER FACTOR           IMPORTANCE   FOREX EQUIVALENT
──────────────────────────────────────────────────────
Close_Price              235        Price momentum (z-score vs MA)
Price_Rate                42        Rate of change (10-bar ROC)
Turnover_Rate             38        Relative volume
Beta                      32        ADX + DI divergence
ROE / ROA                 28        RSI with divergence detection
P/B Ratio                 19        Bollinger Band position
ROS                       15        CCI with slope
The weights in our scoring system come directly from the paper's importance scores, normalized to sum to 1. Momentum gets weight 90. Rate of change gets 75. ADX gets 65. This isn't arbitrary — we're encoding the paper's empirical finding that price behavior dominates everything else.

The scoring engine
Each feature returns a value between 0 and 1, where 1 means strong bullish signal. The final score is a weighted average:

Copy
score = Σ ( weight[i] × feature_score[i] )
Entry LONG  when score ≥ 0.60  AND  6+ features agree (score > 0.55)
Entry SHORT when inverse score ≥ 0.60  AND  6+ features agree
Here's the RSI feature calculation as an example, including divergence detection:

Copy
double rsi  = buf_RSI[1];
double rsip = buf_RSI[2];  // previous bar
double score;
if(rsi < 25.0)      score = 0.82 + (25.0 - rsi) / 100.0;
else if(rsi < 40.0) score = 0.65 + (40.0 - rsi) / 100.0;
else if(rsi < 50.0) score = 0.52;
else if(rsi < 60.0) score = 0.48;
else if(rsi < 75.0) score = 0.35 - (rsi - 60.0) / 100.0;
else                score = 0.18 - (rsi - 75.0) / 100.0;
// Divergence: RSI rising but price falling = bearish divergence
if(rsi > rsip && price < prev_price) score -= 0.05;
if(rsi < rsip && price > prev_price) score += 0.05;
Part 3 — The First Backtest and What Went Wrong
The first version ran on EURUSD H4 over a full year. The results were bad in an instructive way:

Copy
Initial deposit:   $900
Net profit:       -$75.24
Profit Factor:     0.91      ← needs to be > 1.0
Win Rate:         48%
Avg win:          $12.68
Avg loss:         $12.86     ← larger than avg win
Max Drawdown:     23.16%
Sharpe Ratio:    -1.04
A profit factor below 1.0 means losses mathematically exceed gains by definition. With a 48% win rate, you need average wins to exceed average losses by roughly 10% just to break even. Ours were actually smaller. Every single trade had negative expected value.

Three structural problems caused this:

No market regime filter. The EA was entering trades in ranging markets where momentum strategies lose consistently. From August onward the market moved sideways and the system kept firing on weak, ambiguous signals.

Score threshold too low. At 0.55 with only 6 features required to agree, the system was entering on noise. The bar was too low.

The trailing stop was completely broken — three separate bugs, none of which produced any error message in the MT5 journal.

Part 4 — Fixing It
The regime filter
Only trade when the market is actually trending. Simple in concept, transformative in practice:

Copy
void DetectMarketRegime()
{
   double adx     = buf_ADX[1];
   double dip     = buf_DI_P[1];
   double dim     = buf_DI_M[1];
   double di_diff = MathAbs(dip - dim);
   // Trending = ADX above minimum AND DI lines separated
   is_trending = (adx >= 18.0 && di_diff > 4.0);
}
We also added a higher timeframe check. If the daily chart shows a downtrend (price below EMA50, EMA falling, DI- dominant), long entries are blocked regardless of H4 signals. This alone eliminated a large portion of losing trades by stopping the EA from fighting the dominant direction.

The three trailing stop bugs
Bug 1 — Stale ATR between candles. The ATR buffer only updates at the start of each new candle, but ManagePosition runs on every tick. Between candles, buf_ATR[1] was the value from the previous bar — or zero. The fix reads the indicator handle directly on every tick:

Copy
// WRONG — only updates on new candle
double atr = buf_ATR[1];
// CORRECT — fresh read from handle on every tick
double atr_buf[];
ArraySetAsSeries(atr_buf, true);
CopyBuffer(h_ATR, 0, 0, 3, atr_buf);
double atr = atr_buf[1];
Bug 2 — SYMBOL_TRADE_STOPS_LEVEL ignored. Every broker enforces a minimum distance between the current price and any stop loss. If PositionModify places the new SL too close, MT5 silently rejects it — no error, no journal entry, the stop simply doesn't move:

Copy
long   stops_lvl = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
double min_dist  = (stops_lvl + 5) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
// Only modify if new SL respects broker minimum distance
if(new_sl > current_sl && new_sl < bid - min_dist)
   trade.PositionModify(_Symbol, new_sl, tp);
Bug 3 — Breakeven at entry + 2 points. The original code moved the SL to entry + point * 2. Two points. This almost always violated the broker's minimum stop level, causing silent rejection on every single breakeven attempt. Fixed by using min_dist instead.

Part 5 — Capturing the Equity Spikes
After fixing the bugs and adding the regime filter, the EA traded properly. But a new problem appeared in the equity curve: the green line (equity) regularly spiked up 3–5%, then collapsed back. The blue line (balance) barely moved. Profits existed in the equity but never made it to the balance.

The cause was a fixed trailing stop of 1.5×ATR. If the price moves 4×ATR in your direction, a 1.5×ATR trail lets the price retrace 37% of the entire move before stopping out. You watch the equity spike and give almost all of it back.

The solution: a stepped trailing stop that tightens as profit grows.

Copy
Profit < 1×ATR    →   No trailing (let the trade breathe)
Profit 1–2×ATR    →   Trail at 1.2×ATR   (room to run)
Profit 2–3×ATR    →   Trail at 0.8×ATR   (tightening)
Profit > 3×ATR    →   Trail at 0.5×ATR   (lock in gains aggressively)
double profit_atr = (bid - entry_price) / atr;  // for long positions
double trail_dist;
if(profit_atr >= 3.0)
   trail_dist = 0.5 * atr;   // very tight — protect big wins
else if(profit_atr >= 2.0)
   trail_dist = 0.8 * atr;   // medium
else if(profit_atr >= 1.0)
   trail_dist = 1.2 * atr;   // wide — don't get shaken out early
else
   return;  // under 1×ATR — no trailing at all yet
We also added partial position closing. When the trade reaches 2×ATR of profit, 50% of the position closes automatically, converting unrealized equity into realized balance. The remaining half continues with the tight trailing stop.

Copy
if(InpUsePartial && profit_atr >= InpPartial_ATR 
   && partial_done_ticket != current_ticket)
{
   double close_volume = MathFloor(lots * 0.50 / lot_step) * lot_step;
   if(trade.PositionClosePartial(ticket, close_volume))
   {
      partial_done_ticket = current_ticket;  // flag: once per trade
      Print("Partial close executed: ", close_volume, " lots locked in");
   }
}
Part 6 — The Complete Decision Flow
Here is what happens on every H4 candle, in order:

Copy
New H4 candle opens
        │
        ▼
Load all indicator buffers
(RSI, MACD, BB, ATR, ADX, Stochastic, CCI, Volume)
        │
        ▼
Calculate CVaR over last 100 bars
CVaR > 3%? ──YES──► Skip candle, too risky
        │NO
        ▼
Detect LTF market regime via ADX + DI separation
Market ranging? ──YES──► Skip candle
        │NO
        ▼
Check Daily timeframe (EMA50 + ADX)
Assign HTF direction: BULLISH / BEARISH / NEUTRAL
        │
        ▼
Calculate scores for all 10 features [0.0 → 1.0]
Apply importance weights from paper
Compute final LONG score and SHORT score
        │
        ├── LONG score ≥ 0.60 AND 6+ features agree
        │   AND HTF not bearish
        │         └──► Open BUY position
        │
        ├── SHORT score ≥ 0.60 AND 6+ features agree
        │   AND HTF not bullish
        │         └──► Open SELL position
        │
        └── Neither condition met ──► Wait
─────────────────────────────────────────────────────
On EVERY TICK while a position is open:
        │
        ├── Profit ≥ 2×ATR AND partial not yet done?
        │         └──► Close 50% of position
        │
        ├── Profit ≥ 1×ATR AND SL still below entry?
        │         └──► Move SL to breakeven + min_dist
        │
        └── Apply stepped trailing stop
            (distance shrinks as profit grows in ATR units)
The Final Results
With this config

None
After all iterations — regime filter, HTF bias, fixed trailing, partial closes — this is what a full year of EURUSD H4 backtest produced:

Copy
Initial deposit:        $900.00
Net Profit:             $105.66  (+11.74%)
Profit Factor:           1.36
Recovery Factor:         1.21
Sharpe Ratio:            2.79
LR Correlation:          0.87
Total trades:             70
Win Rate:                82.86%
  Longs win rate:        80.36%
  Shorts win rate:       92.86%
Avg profitable trade:    $6.95
Avg losing trade:       -$24.77
Best consecutive run:    12 wins / $104.58
Max Balance Drawdown:    6.60%  ($69.99)
Max Equity Drawdown:     8.15%  ($76.95)
The equity curve over the full year:

None
The balance and equity lines track closely throughout the year — a direct result of the partial closes and stepped trailing converting equity spikes into real balance. The system had a rough February 2025 (the regime filter hadn't been tuned yet at that point in the optimization), recovered through March, and grew steadily from April through July before a normal consolidation period in Q4.

Compare this with the very first backtest from the same dataset:

Copy
FIRST VERSION    FINAL VERSION
────────────────────────────────────────────────
Net Profit           -$75.24         +$105.66
Profit Factor          0.91            1.36
Win Rate              48%             82.86%
Max Drawdown          23.16%          6.60%
Sharpe Ratio          -1.04           2.79
Every single metric improved. The win rate went from coin-flip to 83%. Drawdown dropped from 23% to 6.6%. The Sharpe ratio went from deeply negative to 2.79, which is genuinely strong for a systematic strategy.

The Complete Source Code
Copy
//+------------------------------------------------------------------+
//|                                     FalconEliteAlpha.mq5        |
//|        Based on: "Financial Trading Strategy System Based       |
//|        on Machine Learning" - Chen et al. (2020)                 |
//|  v4.0: Falcon Elite Alpha - Institutional Grade Precision       |
//+------------------------------------------------------------------+
#property copyright   "Falcon Elite Alpha - Institutional Logic"
#property version     "4.00"
#property description "Falcon Elite Alpha: LGBM-Factor Reversion + CVaR"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//|  PARÁMETROS DE ENTRADA                                           |
//+------------------------------------------------------------------+

input group "=== TIMEFRAMES ==="
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_H4; // Timeframe principal de análisis
input ENUM_TIMEFRAMES InpHTF        = PERIOD_D1; // Timeframe mayor (filtro tendencia)
input int             InpWindowSize = 50;        // Ventana de análisis (barras)
input int             InpMinBars    = 150;       // Mínimo barras históricas

input group "=== FILTRO DE RÉGIMEN (reducir si no hay trades) ==="
input bool   InpUseRegimeFilter = true;  // Activar filtro de régimen
input double InpTrend_ADX_Min   = 18.0;  // ADX mínimo tendencia LTF (bajar si no hay trades)
input bool   InpTradeOnlyTrend  = true;  // Solo operar en tendencia
input int    InpHTF_MA_Period   = 50;    // Período EMA en timeframe mayor
input double InpHTF_ADX_Min     = 15.0;  // ADX mínimo HTF (bajar si no hay trades)
input bool   InpHTFNeutralAllow = true;  // Permitir trades cuando HTF es neutro

input group "=== IMPORTANCIA DE FEATURES (Figure 4 del paper) ==="
input int  InpW_Momentum  = 90; // Close_Price equiv. (importancia 235 en paper)
input int  InpW_PriceRate = 75; // Price_Rate equiv.  (importancia 42)
input int  InpW_ADX       = 65; // ADX/Beta equiv.
input int  InpW_MACD      = 60; // MACD
input int  InpW_RSI       = 55; // RSI (ROE/ROA equiv.)
input int  InpW_BB        = 45; // Bollinger (P/B equiv.)
input int  InpW_Stoch     = 35; // Estocástico
input int  InpW_CCI       = 30; // CCI (ROS equiv.)
input int  InpW_ATR       = 25; // ATR volatilidad
input int  InpW_Volume    = 20; // Volumen (Surplus_Reser equiv.)

input group "=== FILTROS DE SEÑAL ==="
input double InpScoreThreshold  = 0.60; // Score mínimo entrada (bajar a 0.58 si no hay trades)
input double InpScoreHysteresis = 0.05; // Histéresis para cierre anticipado
input int    InpMinFeatureAgree = 6;    // Features mínimos alineados

input group "=== PARÁMETROS DE INDICADORES ==="
input int    InpRSI_Period  = 14;
input int    InpMACD_Fast   = 12;
input int    InpMACD_Slow   = 26;
input int    InpMACD_Signal = 9;
input int    InpBB_Period   = 20;
input double InpBB_Dev      = 2.0;
input int    InpATR_Period  = 14;
input int    InpADX_Period  = 14;
input int    InpSTO_K       = 14;
input int    InpSTO_D       = 3;
input int    InpCCI_Period  = 20;
input int    InpROC_Period  = 10;
input int    InpVolMA_Per   = 20;

input group "=== RIESGO CVaR (sección 3.4 del paper) ==="
input int    InpCVaR_Window  = 100;
input double InpConfLevel    = 0.95;
input double InpMaxCVaR_Pct  = 3.0;  // Subido a 3% para permitir más entradas
input double InpMaxRiskPct   = 1.0;
input double InpMaxPortRisk  = 5.0;  // Subido a 5% para portafolio

input group "=== GESTIÓN DE POSICIÓN ==="
input double InpMaxLot       = 1.00;
input bool   InpUseVolSize   = true;
input double InpVolTarget    = 1.0;  // Volatilidad objetivo (%)

input group "=== STOP LOSS / TAKE PROFIT ==="
input double InpSL_ATR_Mult  = 2.0;  // SL inicial en múltiplos de ATR
input double InpTP_ATR_Mult  = 5.0;  // TP en múltiplos de ATR (objetivo máximo)

input group "=== TRAILING ESCALONADO (se aprieta con la ganancia) ==="
// Lógica: cuanto más sube el precio, más apretado es el trailing
// Zona 0 → 1×ATR: sin trailing (dejar respirar)
// Zona 1 → 2×ATR: trailing amplio (1.2×ATR)
// Zona 2 → 3×ATR: trailing medio  (0.8×ATR)
// Zona 3×ATR+  : trailing ajustado(0.5×ATR) para bloquear máximo profit
input bool   InpUseTrailing   = true;
input double InpTrail_Zone1   = 1.2; // Trail ATR cuando profit entre 1-2×ATR
input double InpTrail_Zone2   = 0.8; // Trail ATR cuando profit entre 2-3×ATR
input double InpTrail_Zone3   = 0.5; // Trail ATR cuando profit >3×ATR (muy ajustado)
input bool   InpUseBreakeven  = true;
input double InpBE_ATR        = 1.0; // Activar breakeven al llegar a 1×ATR de ganancia

input group "=== CIERRE PARCIAL (captura ganancias mientras deja correr) ==="
input bool   InpUsePartial    = true;  // Activar cierre parcial
input double InpPartial_ATR   = 2.0;  // Cerrar % al llegar a este ATR de ganancia
input double InpPartialPct    = 50.0; // % del lote a cerrar parcialmente

input group "=== FILTROS DE MERCADO ==="
input bool   InpSpreadFilter    = true;
input int    InpMaxSpread       = 40;  // Subido a 40 puntos
input bool   InpUseTradingHours = false; // DESACTIVADO por defecto para backtest
input int    InpStartHour       = 7;
input int    InpEndHour         = 21;
input bool   InpNoFriday        = false; // DESACTIVADO por defecto para backtest
input int    InpFridayClose     = 20;

input group "=== DIAGNÓSTICO ==="
input bool   InpDiagMode     = true;  // Modo diagnóstico: muestra por qué no entra
input bool   InpDashboard    = true;  // Dashboard en pantalla
input bool   InpDebugLog     = false; // Log detallado en journal

input group "=== CONFIGURACIÓN ==="
input int    InpMagic        = 20240003;
input string InpComment      = "LGB31";

//+------------------------------------------------------------------+
//|  VARIABLES GLOBALES                                              |
//+------------------------------------------------------------------+
CTrade trade;

// Handles
int h_RSI    = INVALID_HANDLE;
int h_MACD   = INVALID_HANDLE;
int h_BB     = INVALID_HANDLE;
int h_ATR    = INVALID_HANDLE;
int h_ADX    = INVALID_HANDLE;
int h_STO    = INVALID_HANDLE;
int h_CCI    = INVALID_HANDLE;
int h_VolMA  = INVALID_HANDLE;
int h_HTF_MA = INVALID_HANDLE;
int h_HTFADX = INVALID_HANDLE;

// Buffers LTF
double buf_RSI[];
double buf_MACD_M[];
double buf_MACD_S[];
double buf_BB_U[];
double buf_BB_M[];
double buf_BB_L[];
double buf_ATR[];
double buf_ADX[];
double buf_DI_P[];
double buf_DI_M[];
double buf_STO_K[];
double buf_STO_D[];
double buf_CCI[];
double buf_Close[];
double buf_High[];
double buf_Low[];
double buf_VolMA[];
long   buf_Volume[];

// Buffers HTF
double buf_HTF_MA[];
double buf_HTF_ADX[];
double buf_HTF_DIP[];
double buf_HTF_DIM[];
double buf_HTF_Close[];

// Pesos y scores
double weights[10];
double feat_scores[10];
string feat_names[10];

// Estado
datetime last_bar     = 0;
double   score_long   = 0.0;
double   score_short  = 0.0;
double   price_ma_g   = 0.0;

// CVaR
double cur_VaR  = 0.0;
double cur_CVaR = 0.0;
double cur_Vol  = 0.0;

// Régimen
bool   is_trending = false;
int    htf_regime  = 0;

// Cierre parcial: registrar si ya se ejecutó para esta posición
ulong  partial_done_ticket = 0; // Ticket de la posición que ya tuvo cierre parcial

// Diagnóstico: razón por la que no entra
string diag_reason = "Iniciando...";

// Stats
int    total_trades = 0;
int    win_trades   = 0;
double total_profit = 0.0;
double max_dd       = 0.0;
double peak_bal     = 0.0;

//+------------------------------------------------------------------+
//|  INIT                                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=== LightGBM_CVaR_EA v3.1 iniciando ===");

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.LogLevel(LOG_LEVEL_NO);

   feat_names[0]="Momentum";    feat_names[1]="PriceRate";
   feat_names[2]="ADX_Trend";   feat_names[3]="MACD";
   feat_names[4]="RSI";         feat_names[5]="BollingerPos";
   feat_names[6]="Stochastic";  feat_names[7]="CCI";
   feat_names[8]="ATR_Rel";     feat_names[9]="Volume";

   double rw[10]={(double)InpW_Momentum,(double)InpW_PriceRate,
                  (double)InpW_ADX,     (double)InpW_MACD,
                  (double)InpW_RSI,     (double)InpW_BB,
                  (double)InpW_Stoch,   (double)InpW_CCI,
                  (double)InpW_ATR,     (double)InpW_Volume};
   double wsum=0.0;
   for(int i=0;i<10;i++) wsum+=rw[i];
   if(wsum<=0.0){Print("ERROR pesos=0"); return INIT_FAILED;}
   for(int i=0;i<10;i++) weights[i]=rw[i]/wsum;

   h_RSI    = iRSI(_Symbol, InpTimeframe, InpRSI_Period, PRICE_CLOSE);
   h_MACD   = iMACD(_Symbol, InpTimeframe, InpMACD_Fast, InpMACD_Slow, InpMACD_Signal, PRICE_CLOSE);
   h_BB     = iBands(_Symbol, InpTimeframe, InpBB_Period, 0, InpBB_Dev, PRICE_CLOSE);
   h_ATR    = iATR(_Symbol, InpTimeframe, InpATR_Period);
   h_ADX    = iADX(_Symbol, InpTimeframe, InpADX_Period);
   h_STO    = iStochastic(_Symbol, InpTimeframe, InpSTO_K, InpSTO_D, 3, MODE_SMA, STO_LOWHIGH);
   h_CCI    = iCCI(_Symbol, InpTimeframe, InpCCI_Period, PRICE_TYPICAL);
   h_VolMA  = iMA(_Symbol, InpTimeframe, InpVolMA_Per, 0, MODE_SMA, VOLUME_TICK);
   h_HTF_MA = iMA(_Symbol, InpHTF, InpHTF_MA_Period, 0, MODE_EMA, PRICE_CLOSE);
   h_HTFADX = iADX(_Symbol, InpHTF, InpADX_Period);

   if(h_RSI==INVALID_HANDLE||h_MACD==INVALID_HANDLE||h_BB==INVALID_HANDLE||
      h_ATR==INVALID_HANDLE||h_ADX==INVALID_HANDLE ||h_STO==INVALID_HANDLE||
      h_CCI==INVALID_HANDLE||h_VolMA==INVALID_HANDLE||
      h_HTF_MA==INVALID_HANDLE||h_HTFADX==INVALID_HANDLE)
     { Print("ERROR: Handle inválido"); return INIT_FAILED; }

   ArraySetAsSeries(buf_RSI,       true);
   ArraySetAsSeries(buf_MACD_M,    true);
   ArraySetAsSeries(buf_MACD_S,    true);
   ArraySetAsSeries(buf_BB_U,      true);
   ArraySetAsSeries(buf_BB_M,      true);
   ArraySetAsSeries(buf_BB_L,      true);
   ArraySetAsSeries(buf_ATR,       true);
   ArraySetAsSeries(buf_ADX,       true);
   ArraySetAsSeries(buf_DI_P,      true);
   ArraySetAsSeries(buf_DI_M,      true);
   ArraySetAsSeries(buf_STO_K,     true);
   ArraySetAsSeries(buf_STO_D,     true);
   ArraySetAsSeries(buf_CCI,       true);
   ArraySetAsSeries(buf_Close,     true);
   ArraySetAsSeries(buf_High,      true);
   ArraySetAsSeries(buf_Low,       true);
   ArraySetAsSeries(buf_VolMA,     true);
   ArraySetAsSeries(buf_Volume,    true);
   ArraySetAsSeries(buf_HTF_MA,    true);
   ArraySetAsSeries(buf_HTF_ADX,   true);
   ArraySetAsSeries(buf_HTF_DIP,   true);
   ArraySetAsSeries(buf_HTF_DIM,   true);
   ArraySetAsSeries(buf_HTF_Close, true);

   peak_bal = AccountInfoDouble(ACCOUNT_BALANCE);
   Print("EA iniciado | Score:", InpScoreThreshold,
         " | ADX_min:", InpTrend_ADX_Min,
         " | HTF_ADX_min:", InpHTF_ADX_Min,
         " | HTFNeutral:", InpHTFNeutralAllow);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//|  DEINIT                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   int handles[10]={h_RSI,h_MACD,h_BB,h_ATR,h_ADX,h_STO,h_CCI,h_VolMA,h_HTF_MA,h_HTFADX};
   for(int i=0;i<10;i++) if(handles[i]!=INVALID_HANDLE) IndicatorRelease(handles[i]);
   Comment("");
   double wr=(total_trades>0)?(double)win_trades/total_trades*100.0:0.0;
   Print("=== EA DETENIDO === Trades:",total_trades," WR:",DoubleToString(wr,1),
         "% P&L:",DoubleToString(total_profit,2)," MaxDD:",DoubleToString(max_dd,2),"%");
  }

//+------------------------------------------------------------------+
//|  TICK PRINCIPAL                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime cur_bar = iTime(_Symbol, InpTimeframe, 0);

   // Gestión continua entre barras
   if(cur_bar == last_bar)
     {
      if(GetMagicTicket()>0) ManagePosition();
      if(InpDashboard) DrawDashboard();
      return;
     }
   last_bar = cur_bar;

   // Verificar barras disponibles
   int bars = iBars(_Symbol, InpTimeframe);
   if(bars < InpMinBars)
     {
      diag_reason = "Esperando datos: " + IntegerToString(bars) + "/" + IntegerToString(InpMinBars);
      Comment(diag_reason);
      return;
     }

   // Cargar datos
   if(!LoadAllData())
     {
      diag_reason = "ERROR: No se pudieron cargar datos de indicadores";
      return;
     }

   // Calcular todos los componentes
   DetectMarketRegime();
   DetectHTFBias();
   CalcCVaR();
   CalcFeatureScores();

   // Scoring ponderado final
   double sl_s=0.0, ss_s=0.0;
   int    al=0, as_c=0;
   for(int i=0;i<10;i++)
     {
      sl_s += weights[i] * feat_scores[i];
      ss_s += weights[i] * (1.0 - feat_scores[i]);
      if(feat_scores[i] > 0.55) al++;
      if(feat_scores[i] < 0.45) as_c++;
     }
   score_long  = sl_s;
   score_short = ss_s;

   // Lógica de entrada / gestión
   bool has_pos = (GetMagicTicket() > 0);

   if(!has_pos)
     {
      bool long_ok  = (sl_s >= InpScoreThreshold && al   >= InpMinFeatureAgree);
      bool short_ok = (ss_s >= InpScoreThreshold && as_c >= InpMinFeatureAgree);

      // Aplicar filtros y registrar razón de bloqueo
      ApplyFilters(long_ok, short_ok);

      if(long_ok)       OpenTrade(ORDER_TYPE_BUY);
      else if(short_ok) OpenTrade(ORDER_TYPE_SELL);
     }
   else
     {
      ManagePosition();
      // Cierre anticipado si señal se invierte fuertemente
      ENUM_POSITION_TYPE pt = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(pt==POSITION_TYPE_BUY  && ss_s>InpScoreThreshold+InpScoreHysteresis && htf_regime==-1)
         trade.PositionClose(_Symbol);
      if(pt==POSITION_TYPE_SELL && sl_s>InpScoreThreshold+InpScoreHysteresis && htf_regime== 1)
         trade.PositionClose(_Symbol);
      diag_reason = "Posición abierta";
     }

   if(InpDashboard) DrawDashboard();
  }

//+------------------------------------------------------------------+
//|  APLICAR FILTROS CON DIAGNÓSTICO                                |
//+------------------------------------------------------------------+
void ApplyFilters(bool &long_ok, bool &short_ok)
  {
   // Reset razón de diagnóstico
   diag_reason = "OK - esperando señal";

   // --- Filtro 1: Score insuficiente ---
   if(!long_ok && !short_ok)
     {
      diag_reason = StringFormat("Score bajo: L=%.4f S=%.4f (min %.2f) | AgreL=%d AgreS=%d (min %d)",
                                 score_long, score_short, InpScoreThreshold,
                                 /* recalcular agree */ 0, 0, InpMinFeatureAgree);
      // Recalcular agree para diagnóstico
      int al=0, as_c=0;
      for(int i=0;i<10;i++){if(feat_scores[i]>0.55)al++; if(feat_scores[i]<0.45)as_c++;}
      diag_reason = StringFormat("Score bajo: L=%.4f(%d) S=%.4f(%d) umbral=%.2f/%d",
                                 score_long,al,score_short,as_c,InpScoreThreshold,InpMinFeatureAgree);
      return;
     }

   // --- Filtro 2: Régimen de mercado ---
   if(InpUseRegimeFilter && InpTradeOnlyTrend && !is_trending)
     {
      diag_reason = StringFormat("FILTRO RÉGIMEN: ADX=%.1f (min %.1f) - Mercado en rango",
                                 buf_ADX[1], InpTrend_ADX_Min);
      long_ok = false;
      short_ok = false;
      return;
     }

   // --- Filtro 3: HTF ---
   if(htf_regime == 1 && short_ok)
     {
      diag_reason = "FILTRO HTF: HTF alcista, cancelando SHORT";
      short_ok = false;
     }
   if(htf_regime == -1 && long_ok)
     {
      diag_reason = "FILTRO HTF: HTF bajista, cancelando LONG";
      long_ok = false;
     }
   if(htf_regime == 0 && !InpHTFNeutralAllow)
     {
      diag_reason = StringFormat("FILTRO HTF: HTF neutro (ADX HTF=%.1f min=%.1f)",
                                 buf_HTF_ADX[1], InpHTF_ADX_Min);
      long_ok = false;
      short_ok = false;
      return;
     }

   // --- Filtro 4: CVaR ---
   if(cur_CVaR * 100.0 > InpMaxCVaR_Pct)
     {
      diag_reason = StringFormat("FILTRO CVaR: %.3f%% > max %.1f%%",
                                 cur_CVaR*100.0, InpMaxCVaR_Pct);
      long_ok = false;
      short_ok = false;
      return;
     }

   // --- Filtro 5: Spread ---
   if(InpSpreadFilter)
     {
      long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(spread > InpMaxSpread)
        {
         diag_reason = StringFormat("FILTRO SPREAD: %d > max %d", (int)spread, InpMaxSpread);
         long_ok = false;
         short_ok = false;
         return;
        }
     }

   // --- Filtro 6: Horario ---
   if(InpUseTradingHours)
     {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(InpNoFriday && dt.day_of_week==5 && dt.hour>=InpFridayClose)
        {
         CloseAllMagicPositions();
         diag_reason = "FILTRO HORARIO: Cierre viernes";
         long_ok=false; short_ok=false;
         return;
        }
      if(dt.hour < InpStartHour || dt.hour >= InpEndHour)
        {
         diag_reason = StringFormat("FILTRO HORARIO: Hora %d fuera de [%d-%d]",
                                    dt.hour, InpStartHour, InpEndHour);
         long_ok=false; short_ok=false;
         return;
        }
     }

   // --- Filtro 7: Riesgo portafolio ---
   double port_risk = CalcPortfolioRisk();
   if(port_risk + InpMaxRiskPct/100.0 > InpMaxPortRisk/100.0)
     {
      diag_reason = StringFormat("FILTRO RIESGO PORTA: %.2f%% + %.1f%% > max %.1f%%",
                                 port_risk*100, InpMaxRiskPct, InpMaxPortRisk);
      long_ok=false; short_ok=false;
      return;
     }

   // Si llegamos aquí hay señal válida
   if(long_ok)  diag_reason = StringFormat(">>> SEÑAL LONG  score=%.4f agree=%d", score_long, 0);
   if(short_ok) diag_reason = StringFormat(">>> SEÑAL SHORT score=%.4f agree=%d", score_short, 0);
  }

//+------------------------------------------------------------------+
//|  CARGAR DATOS                                                    |
//+------------------------------------------------------------------+
bool LoadAllData()
  {
   int n = MathMax(InpWindowSize, InpCVaR_Window) + 20;

   if(CopyClose(_Symbol,    InpTimeframe, 0, n, buf_Close)   < n) return false;
   if(CopyHigh(_Symbol,     InpTimeframe, 0, n, buf_High)    < n) return false;
   if(CopyLow(_Symbol,      InpTimeframe, 0, n, buf_Low)     < n) return false;
   if(CopyTickVolume(_Symbol,InpTimeframe,0, n, buf_Volume)  < n) return false;

   if(CopyBuffer(h_RSI,   0,0,n,buf_RSI)    < n) return false;
   if(CopyBuffer(h_MACD,  0,0,n,buf_MACD_M) < n) return false;
   if(CopyBuffer(h_MACD,  1,0,n,buf_MACD_S) < n) return false;
   if(CopyBuffer(h_BB,    1,0,n,buf_BB_U)   < n) return false;
   if(CopyBuffer(h_BB,    0,0,n,buf_BB_M)   < n) return false;
   if(CopyBuffer(h_BB,    2,0,n,buf_BB_L)   < n) return false;
   if(CopyBuffer(h_ATR,   0,0,n,buf_ATR)    < n) return false;
   if(CopyBuffer(h_ADX,   0,0,n,buf_ADX)    < n) return false;
   if(CopyBuffer(h_ADX,   1,0,n,buf_DI_P)   < n) return false;
   if(CopyBuffer(h_ADX,   2,0,n,buf_DI_M)   < n) return false;
   if(CopyBuffer(h_STO,   0,0,n,buf_STO_K)  < n) return false;
   if(CopyBuffer(h_STO,   1,0,n,buf_STO_D)  < n) return false;
   if(CopyBuffer(h_CCI,   0,0,n,buf_CCI)    < n) return false;
   if(CopyBuffer(h_VolMA, 0,0,n,buf_VolMA)  < n) return false;

   if(CopyBuffer(h_HTF_MA,  0,0,10,buf_HTF_MA)   < 5) return false;
   if(CopyBuffer(h_HTFADX,  0,0,10,buf_HTF_ADX)  < 5) return false;
   if(CopyBuffer(h_HTFADX,  1,0,10,buf_HTF_DIP)  < 5) return false;
   if(CopyBuffer(h_HTFADX,  2,0,10,buf_HTF_DIM)  < 5) return false;
   if(CopyClose(_Symbol, InpHTF, 0, 10, buf_HTF_Close) < 5) return false;

   return true;
  }

//+------------------------------------------------------------------+
//|  DETECTAR RÉGIMEN DE MERCADO                                    |
//+------------------------------------------------------------------+
void DetectMarketRegime()
  {
   double adx  = buf_ADX[1];
   double dip  = buf_DI_P[1];
   double dim  = buf_DI_M[1];
   is_trending = (adx >= InpTrend_ADX_Min && MathAbs(dip-dim) > 4.0);
  }

//+------------------------------------------------------------------+
//|  DETECTAR BIAS DEL TIMEFRAME MAYOR                              |
//+------------------------------------------------------------------+
void DetectHTFBias()
  {
   if(ArraySize(buf_HTF_MA)<5 || ArraySize(buf_HTF_Close)<5)
     { htf_regime=0; return; }

   double cl  = buf_HTF_Close[1];
   double man = buf_HTF_MA[1];
   double map = buf_HTF_MA[3];
   double adx = buf_HTF_ADX[1];
   double dip = buf_HTF_DIP[1];
   double dim = buf_HTF_DIM[1];

   // Si ADX HTF es muy bajo → neutro
   if(adx < InpHTF_ADX_Min)
     { htf_regime=0; return; }

   bool above = (cl > man);
   bool rising= (man > map);

   if(above  && rising  && dip>dim) htf_regime =  1;
   else if(!above && !rising && dim>dip) htf_regime = -1;
   else htf_regime = 0;
  }

//+------------------------------------------------------------------+
//|  CALCULAR CVaR Y VaR                                            |
//+------------------------------------------------------------------+
void CalcCVaR()
  {
   int n=MathMin(InpCVaR_Window, ArraySize(buf_Close)-2);
   if(n<20) return;

   double rets[]; ArrayResize(rets,n);
   for(int i=0;i<n;i++)
      rets[i]=(buf_Close[i+2]>0.0)?MathLog(buf_Close[i+1]/buf_Close[i+2]):0.0;

   double mean=0.0;
   for(int i=0;i<n;i++) mean+=rets[i];
   mean/=n;

   double vsum=0.0;
   for(int i=0;i<n;i++) vsum+=MathPow(rets[i]-mean,2);
   double sd=MathSqrt(vsum/n);
   cur_Vol=sd;

   double z=(InpConfLevel>=0.99)?2.326:(InpConfLevel>=0.975)?1.960:1.645;
   cur_VaR=-(mean-z*sd);

   double phi=MathExp(-0.5*z*z)/MathSqrt(2.0*M_PI);
   double pc =-(mean-sd*phi/(1.0-InpConfLevel));

   double rs[]; ArrayCopy(rs,rets); ArraySort(rs);
   int tail=MathMax(1,(int)MathFloor(n*(1.0-InpConfLevel)));
   double hsum=0.0;
   for(int i=0;i<tail&&i<n;i++) hsum+=rs[i];
   double hc=-hsum/tail;

   cur_CVaR=(pc+hc)/2.0;
   if(cur_CVaR<0.0) cur_CVaR=MathAbs(cur_CVaR);
  }

//+------------------------------------------------------------------+
//|  CALCULAR SCORES DE FEATURES                                    |
//+------------------------------------------------------------------+
void CalcFeatureScores()
  {
   // FEATURE 0: Momentum (Close_Price - importancia 235 en paper)
   double pma=0.0;
   for(int i=1;i<=InpWindowSize;i++) pma+=buf_Close[i];
   pma/=InpWindowSize;
   price_ma_g=pma;

   double pstd=0.0;
   for(int i=1;i<=InpWindowSize;i++) pstd+=MathPow(buf_Close[i]-pma,2);
   pstd=MathSqrt(pstd/InpWindowSize);
   feat_scores[0]=Sigmoid((pstd>0)?(buf_Close[1]-pma)/pstd:0.0);

   // FEATURE 1: Rate of Change (Price_Rate - importancia 42 en paper)
   double roc=0.0;
   int roc_idx=InpROC_Period+1;
   if(roc_idx<ArraySize(buf_Close) && buf_Close[roc_idx]>0.0)
      roc=(buf_Close[1]-buf_Close[roc_idx])/buf_Close[roc_idx]*100.0;
   feat_scores[1]=Sigmoid(roc*2.5);

   // FEATURE 2: ADX + DI divergencia (Beta/tendencia)
   double adx=buf_ADX[1], dip=buf_DI_P[1], dim=buf_DI_M[1];
   double adx_s=0.5;
   if(adx>=InpTrend_ADX_Min && (dip+dim)>0.0)
      adx_s=0.5+(dip-dim)/(dip+dim)*0.4*MathMin(adx/50.0,1.0);
   feat_scores[2]=MathMax(0.0,MathMin(1.0,adx_s));

   // FEATURE 3: MACD histograma
   double mh =buf_MACD_M[1]-buf_MACD_S[1];
   double mhp=buf_MACD_M[2]-buf_MACD_S[2];
   double ms;
   if(mh>0.0&&mhp<=0.0)      ms=0.82;
   else if(mh<0.0&&mhp>=0.0) ms=0.18;
   else if(mh>0.0&&mh>mhp)   ms=0.65;
   else if(mh<0.0&&mh<mhp)   ms=0.35;
   else if(buf_MACD_M[1]>0.0) ms=0.58;
   else if(buf_MACD_M[1]<0.0) ms=0.42;
   else                        ms=0.50;
   feat_scores[3]=ms;

   // FEATURE 4: RSI + divergencias
   double rsi=buf_RSI[1], rsip=buf_RSI[2];
   double rs;
   if(rsi<25.0)       rs=0.82+(25.0-rsi)/100.0;
   else if(rsi<40.0)  rs=0.65+(40.0-rsi)/100.0;
   else if(rsi<50.0)  rs=0.52;
   else if(rsi<60.0)  rs=0.48;
   else if(rsi<75.0)  rs=0.35-(rsi-60.0)/100.0;
   else               rs=0.18-(rsi-75.0)/100.0;
   if(rsi>rsip && buf_Close[1]<buf_Close[2]) rs-=0.05;
   if(rsi<rsip && buf_Close[1]>buf_Close[2]) rs+=0.05;
   feat_scores[4]=MathMax(0.0,MathMin(1.0,rs));

   // FEATURE 5: Posición Bollinger + squeeze
   double bw=buf_BB_U[1]-buf_BB_L[1];
   double bwma=0.0;
   for(int i=1;i<=20&&i<ArraySize(buf_BB_U);i++) bwma+=(buf_BB_U[i]-buf_BB_L[i]);
   bwma/=20.0;
   double bp=(bw>0.0)?(buf_Close[1]-buf_BB_L[1])/bw:0.5;
   double bs;
   if(bp<0.15)      bs=0.80;
   else if(bp<0.30) bs=0.65;
   else if(bp<0.45) bs=0.55;
   else if(bp<0.55) bs=0.50;
   else if(bp<0.70) bs=0.45;
   else if(bp<0.85) bs=0.35;
   else             bs=0.20;
   if(bw>bwma*1.2){ if(bp>0.5) bs=MathMin(bs+0.06,1.0); else bs=MathMax(bs-0.06,0.0); }
   feat_scores[5]=MathMax(0.0,MathMin(1.0,bs));

   // FEATURE 6: Estocástico
   double sk=buf_STO_K[1], sdv=buf_STO_D[1];
   double skp=buf_STO_K[2], sdvp=buf_STO_D[2];
   double sts;
   if(sk<20.0&&sdv<20.0&&sk>sdv&&skp<=sdvp)      sts=0.85;
   else if(sk>80.0&&sdv>80.0&&sk<sdv&&skp>=sdvp)  sts=0.15;
   else if(sk<25.0&&sdv<25.0) sts=0.68;
   else if(sk>75.0&&sdv>75.0) sts=0.32;
   else if(sk>sdv&&sk>skp)    sts=0.60;
   else if(sk<sdv&&sk<skp)    sts=0.40;
   else                        sts=0.50;
   feat_scores[6]=sts;

   // FEATURE 7: CCI
   double cci=buf_CCI[1], ccip=buf_CCI[2];
   double cs;
   if(cci<-150.0)      cs=0.82;
   else if(cci<-75.0)  cs=0.65;
   else if(cci<-25.0)  cs=0.55;
   else if(cci<25.0)   cs=0.50;
   else if(cci<75.0)   cs=0.45;
   else if(cci<150.0)  cs=0.35;
   else                cs=0.18;
   if(cci>ccip&&cci<0.0)  cs+=0.05;
   if(cci<ccip&&cci>0.0)  cs-=0.05;
   feat_scores[7]=MathMax(0.0,MathMin(1.0,cs));

   // FEATURE 8: ATR relativo
   double atr_ma=0.0;
   for(int i=1;i<=InpWindowSize&&i<ArraySize(buf_ATR);i++) atr_ma+=buf_ATR[i];
   atr_ma/=InpWindowSize;
   double ar=(atr_ma>0.0)?buf_ATR[1]/atr_ma:1.0;
   double ats;
   if(ar<0.50)      ats=0.35;
   else if(ar<0.80) ats=0.45;
   else if(ar<1.30) ats=0.55;
   else if(ar<2.00) ats=0.50;
   else             ats=0.38;
   ats+=(buf_Close[1]>pma)?0.05:-0.05;
   feat_scores[8]=MathMax(0.0,MathMin(1.0,ats));

   // FEATURE 9: Volumen relativo
   double vr=(buf_VolMA[1]>0.0)?(double)buf_Volume[1]/buf_VolMA[1]:1.0;
   double vs;
   if(vr>2.0)       vs=(buf_Close[1]>buf_Close[2])?0.78:0.22;
   else if(vr>1.4)  vs=(buf_Close[1]>buf_Close[2])?0.65:0.35;
   else if(vr>1.0)  vs=(buf_Close[1]>buf_Close[2])?0.57:0.43;
   else             vs=0.50;
   feat_scores[9]=vs;
  }

//+------------------------------------------------------------------+
//|  CALCULAR TAMAÑO DE POSICIÓN                                    |
//+------------------------------------------------------------------+
double CalcLotSize(double sl_dist)
  {
   double bal      = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq       = AccountInfoDouble(ACCOUNT_EQUITY);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_sz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(sl_dist<=0.0||tick_val<=0.0||tick_sz<=0.0) return min_lot;

   double lots=0.0;
   if(InpUseVolSize && cur_Vol>0.0)
     {
      double vtd  = InpVolTarget/100.0;
      double cont = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
      double pval = cont*buf_Close[1];
      if(pval>0.0) lots=(eq*vtd)/(cur_Vol*pval);
      if(cur_CVaR>0.0) lots*=MathMin(1.0,(InpMaxCVaR_Pct/100.0)/cur_CVaR);
     }
   else
     {
      double risk=bal*InpMaxRiskPct/100.0;
      double rpl =(sl_dist/tick_sz)*tick_val;
      if(rpl>0.0) lots=risk/rpl;
     }

   lots=MathFloor(lots/lot_step)*lot_step;
   lots=MathMax(lots,min_lot);
   lots=MathMin(lots,InpMaxLot);

   double margin=0.0;
   if(OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,lots,SymbolInfoDouble(_Symbol,SYMBOL_ASK),margin))
      if(margin>eq*0.40&&margin>0.0)
        {
         lots=MathFloor(eq*0.40/margin*lots/lot_step)*lot_step;
         lots=MathMax(lots,min_lot);
        }
   return lots;
  }

//+------------------------------------------------------------------+
//|  ABRIR OPERACIÓN                                                 |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE ot)
  {
   double atr=buf_ATR[1];
   if(atr<=0.0){Print("ERROR ATR=0"); return;}

   double ask   =SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid   =SymbolInfoDouble(_Symbol,SYMBOL_BID);
   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double sl_d  =InpSL_ATR_Mult*atr;
   double tp_d  =InpTP_ATR_Mult*atr;


   double ep,sl,tp;
   if(ot==ORDER_TYPE_BUY)
     {ep=ask; sl=NormalizeDouble(ask-sl_d,digits); tp=NormalizeDouble(ask+tp_d,digits);}
   else
     {ep=bid; sl=NormalizeDouble(bid+sl_d,digits); tp=NormalizeDouble(bid-tp_d,digits);}

   double lots=CalcLotSize(sl_d);
   string cmt=InpComment+(ot==ORDER_TYPE_BUY?"_L":"_S")
             +"_A"+DoubleToString(buf_ADX[1],0)
             +"_H"+(htf_regime==1?"U":htf_regime==-1?"D":"N");

   bool ok=(ot==ORDER_TYPE_BUY)?
            trade.Buy(lots,_Symbol,ep,sl,tp,cmt):
            trade.Sell(lots,_Symbol,ep,sl,tp,cmt);

   if(ok)
     {
      total_trades++;
      diag_reason="Posición abierta";
      Print(">>> TRADE ",EnumToString(ot)," lots=",lots," sl=",sl," tp=",tp,
            " score=",DoubleToString(ot==ORDER_TYPE_BUY?score_long:score_short,4),
            " CVaR=",DoubleToString(cur_CVaR*100,2),"% HTF=",htf_regime);
     }
   else
      Print("ERROR trade: ",trade.ResultRetcodeDescription()," (",trade.ResultRetcode(),")");
  }

//+------------------------------------------------------------------+
//|  GESTIONAR POSICIÓN ABIERTA (Trailing + Breakeven robustos)     |
//+------------------------------------------------------------------+
void ManagePosition()
  {
   ulong tkt=GetMagicTicket();
   if(tkt==0||!PositionSelectByTicket(tkt)) return;

   double sl0  = PositionGetDouble(POSITION_SL);
   double tp0  = PositionGetDouble(POSITION_TP);
   double ep   = PositionGetDouble(POSITION_PRICE_OPEN);
   double lts  = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   double bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   int    dg  = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double pnt = SymbolInfoDouble(_Symbol,SYMBOL_POINT);

   // Stop level mínimo del broker
   long   stops_lvl = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double min_dist  = (stops_lvl+5)*pnt;

   // ATR fresco del handle (no usar buf_ATR que se actualiza solo en nueva barra)
   double atr_buf[]; ArraySetAsSeries(atr_buf,true);
   if(CopyBuffer(h_ATR,0,0,3,atr_buf)<3) return;
   double atr = atr_buf[1];
   if(atr<=0.0) return;

   // === PRECIO DE REFERENCIA Y GANANCIA ACTUAL ===
   double precio_ref = (pt==POSITION_TYPE_BUY) ? bid : ask;
   double profit_pts = (pt==POSITION_TYPE_BUY) ? (bid-ep) : (ep-ask);
   // profit en múltiplos de ATR
   double profit_atr = profit_pts / atr;

   // === CIERRE PARCIAL ===
   // Se ejecuta UNA sola vez por posición al llegar a InpPartial_ATR de ganancia
   if(InpUsePartial && profit_atr >= InpPartial_ATR && partial_done_ticket != tkt)
     {
      double lot_step  = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
      double min_lot   = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
      double close_vol = MathFloor(lts*(InpPartialPct/100.0)/lot_step)*lot_step;

      if(close_vol >= min_lot && close_vol < lts)
        {
         if(trade.PositionClosePartial(tkt, close_vol))
           {
            partial_done_ticket = tkt; // Marcar como ejecutado para esta posición
            Print(">> Cierre parcial: ",close_vol," lotes de ",lts,
                  " | Profit ATR=",DoubleToString(profit_atr,2),
                  " | Lote restante=",DoubleToString(lts-close_vol,2));
           }
         else
            Print("!! Error cierre parcial: ",trade.ResultRetcodeDescription());
        }
      else
         partial_done_ticket = tkt; // Lote demasiado pequeño para dividir, marcar igual
      return; // No modificar SL el mismo tick del cierre parcial
     }

   // === BREAKEVEN (prioridad sobre trailing) ===
   if(InpUseBreakeven && profit_atr >= InpBE_ATR)
     {
      if(pt==POSITION_TYPE_BUY && sl0 < ep)
        {
         double nsl = NormalizeDouble(ep + min_dist, dg);
         if(nsl > sl0 && nsl < bid - min_dist)
           {
            if(!trade.PositionModify(_Symbol,nsl,tp0))
               Print("!! BE BUY error: ",trade.ResultRetcodeDescription());
            else
               Print(">> BE BUY activado: SL ",DoubleToString(sl0,dg),"->",DoubleToString(nsl,dg));
            return; // No trailing en mismo tick
           }
        }
      else if(pt==POSITION_TYPE_SELL && sl0 > ep)
        {
         double nsl = NormalizeDouble(ep - min_dist, dg);
         if(nsl < sl0 && nsl > ask + min_dist)
           {
            if(!trade.PositionModify(_Symbol,nsl,tp0))
               Print("!! BE SELL error: ",trade.ResultRetcodeDescription());
            else
               Print(">> BE SELL activado: SL ",DoubleToString(sl0,dg),"->",DoubleToString(nsl,dg));
            return;
           }
        }
     }

   // === TRAILING ESCALONADO ===
   // La distancia del trailing se reduce conforme sube el profit
   // Esto permite capturar más ganancia en los spikes del equity
   if(!InpUseTrailing) return;

   double trail_dist;
   if(profit_atr >= 3.0)
      trail_dist = InpTrail_Zone3 * atr; // > 3×ATR: muy ajustado, bloquear ganancias
   else if(profit_atr >= 2.0)
      trail_dist = InpTrail_Zone2 * atr; // 2-3×ATR: medio
   else if(profit_atr >= 1.0)
      trail_dist = InpTrail_Zone1 * atr; // 1-2×ATR: amplio
   else
      return; // < 1×ATR: no trailing todavía, dejar respirar

   // Asegurar que trail_dist respeta stop level del broker
   trail_dist = MathMax(trail_dist, min_dist * 1.5);

   if(pt==POSITION_TYPE_BUY)
     {
      double nsl = NormalizeDouble(bid - trail_dist, dg);
      if(nsl > sl0 && nsl < bid - min_dist)
        {
         if(!trade.PositionModify(_Symbol,nsl,tp0))
            Print("!! Trail BUY error: ",trade.ResultRetcodeDescription(),
                  " nsl=",nsl," sl0=",sl0," bid=",bid," atr_zona=",DoubleToString(profit_atr,1));
        }
     }
   else if(pt==POSITION_TYPE_SELL)
     {
      double nsl = NormalizeDouble(ask + trail_dist, dg);
      if(nsl < sl0 && nsl > ask + min_dist)
        {
         if(!trade.PositionModify(_Symbol,nsl,tp0))
            Print("!! Trail SELL error: ",trade.ResultRetcodeDescription(),
                  " nsl=",nsl," sl0=",sl0," ask=",ask," atr_zona=",DoubleToString(profit_atr,1));
        }
     }
  }

//+------------------------------------------------------------------+
//|  CALCULAR RIESGO PORTAFOLIO                                     |
//+------------------------------------------------------------------+
double CalcPortfolioRisk()
  {
   double bal=AccountInfoDouble(ACCOUNT_BALANCE); if(bal<=0.0) return 0.0;
   double risk=0.0;
   for(int i=0;i<PositionsTotal();i++)
     {
      ulong tkt=PositionGetTicket(i);
      if(!PositionSelectByTicket(tkt)) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=InpMagic) continue;
      double op=PositionGetDouble(POSITION_PRICE_OPEN);
      double sl=PositionGetDouble(POSITION_SL);
      double lts=PositionGetDouble(POSITION_VOLUME);
      if(sl==0.0) continue;
      string sym=PositionGetString(POSITION_SYMBOL);
      double tv=SymbolInfoDouble(sym,SYMBOL_TRADE_TICK_VALUE);
      double ts=SymbolInfoDouble(sym,SYMBOL_TRADE_TICK_SIZE);
      if(ts>0.0&&tv>0.0) risk+=(MathAbs(op-sl)/ts)*tv*lts/bal;
     }
   return risk;
  }

//+------------------------------------------------------------------+
//|  OBTENER TICKET                                                  |
//+------------------------------------------------------------------+
ulong GetMagicTicket()
  {
   for(int i=0;i<PositionsTotal();i++)
     {
      ulong tkt=PositionGetTicket(i);
      if(PositionSelectByTicket(tkt))
         if(PositionGetString(POSITION_SYMBOL)==_Symbol&&
            PositionGetInteger(POSITION_MAGIC)==InpMagic) return tkt;
     }
   return 0;
  }

//+------------------------------------------------------------------+
//|  CERRAR POSICIONES                                              |
//+------------------------------------------------------------------+
void CloseAllMagicPositions()
  {
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong tkt=PositionGetTicket(i);
      if(PositionSelectByTicket(tkt))
         if(PositionGetString(POSITION_SYMBOL)==_Symbol&&
            PositionGetInteger(POSITION_MAGIC)==InpMagic) trade.PositionClose(tkt);
     }
  }

//+------------------------------------------------------------------+
//|  SIGMOIDE                                                        |
//+------------------------------------------------------------------+
double Sigmoid(double x){ return 1.0/(1.0+MathExp(-x)); }

//+------------------------------------------------------------------+
//|  EVENTO TRADE CERRADO                                           |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult  &res)
  {
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal,DEAL_MAGIC)!=InpMagic) return;
   ENUM_DEAL_ENTRY de=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
   if(de!=DEAL_ENTRY_OUT&&de!=DEAL_ENTRY_OUT_BY) return;

   double pnl =HistoryDealGetDouble(trans.deal,DEAL_PROFIT);
   double swap=HistoryDealGetDouble(trans.deal,DEAL_SWAP);
   double comm=HistoryDealGetDouble(trans.deal,DEAL_COMMISSION);
   double net =pnl+swap+comm;
   total_profit+=net;
   if(pnl>0.0) win_trades++;
   // Resetear estado de cierre parcial para la próxima posición
   partial_done_ticket=0;
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal>peak_bal) peak_bal=bal;
   double dd=(peak_bal>0.0)?(peak_bal-bal)/peak_bal*100.0:0.0;
   if(dd>max_dd) max_dd=dd;
   double wr=(total_trades>0)?(double)win_trades/total_trades*100.0:0.0;
   Print("<<< CERRADO PnL:",DoubleToString(pnl,2)," Net:",DoubleToString(net,2),
         " Total:",DoubleToString(total_profit,2)," WR:",DoubleToString(wr,1),"%");
  }

//+------------------------------------------------------------------+
//|  DASHBOARD                                                       |
//+------------------------------------------------------------------+
void DrawDashboard()
  {
   double wr=(total_trades>0)?(double)win_trades/total_trades*100.0:0.0;
   string reg_s=is_trending?(buf_DI_P[1]>buf_DI_M[1]?"TREND UP":"TREND DWN"):"RANGE";
   string htf_s=(htf_regime==1)?"▲ ALCISTA":(htf_regime==-1)?"▼ BAJISTA":"◆ NEUTRO";

   // Calcular agree para mostrar
   int al=0,as_c=0;
   for(int i=0;i<10;i++){if(feat_scores[i]>0.55)al++; if(feat_scores[i]<0.45)as_c++;}

   string d="╔══════════════════════════════════════════╗\n";
   d+="║  LightGBM+CVaR EA v3.1 | Chen 2020      ║\n";
   d+="╠══════════════════════════════════════════╣\n";
   d+="║ LTF: "+StringFormat("%-10s",reg_s)+" HTF: "+StringFormat("%-12s",htf_s)+"║\n";
   d+="║ ADX="+StringFormat("%-5.1f",buf_ADX[1])+" DI+="+StringFormat("%-5.1f",buf_DI_P[1])+
      " DI-="+StringFormat("%-5.1f",buf_DI_M[1])+"                ║\n";
   d+="╠══════════════════════════════════════════╣\n";
   d+="║ SCORE LONG:  "+StringFormat("%.4f",score_long)+" ("+IntegerToString(al)+"/"+IntegerToString(InpMinFeatureAgree)+")"+
      (score_long>=InpScoreThreshold?"  [✓]  ":"       ")+"       ║\n";
   d+="║ SCORE SHORT: "+StringFormat("%.4f",score_short)+" ("+IntegerToString(as_c)+"/"+IntegerToString(InpMinFeatureAgree)+")"+
      (score_short>=InpScoreThreshold?"  [✓]  ":"       ")+"       ║\n";
   d+="╠══════════════════════════════════════════╣\n";
   d+="║ VaR:"+StringFormat("%5.3f",cur_VaR*100)+" CVaR:"+StringFormat("%5.3f",cur_CVaR*100)+" Vol:"+StringFormat("%6.4f",cur_Vol*100)+"% ║\n";
   d+="╠══════════════════════════════════════════╣\n";

   // Features con barra visual
   for(int i=0;i<10;i++)
     {
      int f=(int)MathRound(feat_scores[i]*10.0);
      string bar=""; for(int b=0;b<10;b++) bar+=(b<f?"█":"░");
      string marker=(feat_scores[i]>0.55)?"▲":(feat_scores[i]<0.45)?"▼":"─";
      d+="║"+marker+" "+StringFormat("%-13s",feat_names[i])+" "+bar+" ║\n";
     }

   d+="╠══════════════════════════════════════════╣\n";
   if(InpDiagMode)
      d+="║ "+StringFormat("%-42s",diag_reason)+"║\n";
   d+="╠══════════════════════════════════════════╣\n";
   d+="║ Trades:"+StringFormat("%-4d",total_trades)+" WR:"+StringFormat("%-5.1f",wr)+
      "% P&L:"+StringFormat("%-10.2f",total_profit)+"     ║\n";
   d+="║ MaxDD: "+StringFormat("%-6.2f",max_dd)+"%                         ║\n";
   d+="╚══════════════════════════════════════════╝";
   Comment(d);
  }

//+------------------------------------------------------------------+
//|  OPTIMIZADOR                                                     |
//+------------------------------------------------------------------+
double OnTester()
  {
   double profit=TesterStatistics(STAT_PROFIT);
   double dd    =TesterStatistics(STAT_EQUITY_DD_RELATIVE);
   double trades=TesterStatistics(STAT_TRADES);
   double sharpe=TesterStatistics(STAT_SHARPE_RATIO);
   double pf    =TesterStatistics(STAT_PROFIT_FACTOR);
   Print("=== BACKTEST v3.1 === Profit:",DoubleToString(profit,2),
         " DD:",DoubleToString(dd,2),"% Trades:",(int)trades,
         " Sharpe:",DoubleToString(sharpe,3)," PF:",DoubleToString(pf,3));
   if(trades<15.0||sharpe<=0.0||pf<=1.0) return 0.0;
   return sharpe*MathMin(pf,3.0)*(1.0-dd/100.0);
  }
//+------------------------------------------------------------------+
Part 7 — What I'd Do Differently
A few honest reflections:

The CVaR calculation assumes roughly normal returns, which forex returns aren't — they have fat tails, especially around news events. A more robust implementation would use a longer lookback window or apply an extreme value distribution to the tail. The hybrid approach used here (average of parametric and historical CVaR) is a reasonable first step but not the final answer.

The feature weights are static. In the original paper, LightGBM retrains every quarter on fresh data, so importance weights naturally shift with market conditions. Our weights are fixed at initialization. An obvious improvement would be recalculating weights periodically based on recent signal performance — a lightweight online learning layer.

The regime filter is binary. Either the market is trending or it isn't. A more nuanced approach would scale position size continuously with regime confidence — a market with ADX at 40 and DI separation of 20 points is much more clearly trending than one with ADX at 20 and DI separation of 5. Continuous scaling rather than binary on/off would likely improve consistency.

Finally, this system trades one pair. The paper's portfolio construction layer allocates across multiple assets simultaneously to reduce correlation risk. Extending this to a basket of forex pairs with genuine minimum variance allocation across them is the most significant remaining piece of work — and the subject of a future article.

Conclusion
The paper by Chen et al. is genuinely useful — not because you can port its code to MT5 and run it, but because its underlying framework is sound and translatable. The insight that price momentum dominates fundamental factors (confirmed by the importance scores in Figure 4) is directly actionable even with no balance sheet data. The CVaR framework is more rigorous than most retail position sizing methods. And the minimum variance approach provably beats equal-weighting when volatility differs between instruments.

The journey from paper to working EA involved more debugging than modeling. Three silent bugs in the trailing stop. A regime filter that was first too tight (zero trades), then too loose (too many losers). A threshold calibration that took multiple backtest iterations to land correctly.

That debugging process is, in my view, the most transferable part of this project. Academic papers present clean results. Real implementations live in the messy space between clean theory and the broker's minimum stop level requirement at 3 a.m. on a Thursday.

The code is above. Build it, break it, improve it.

Enjoyed this? The next article will cover adding genuine online retraining using MT5's Python integration — getting closer to the actual LightGBM model the paper describes.

#learning
#programming
#research
#forex
#cryptocurrency




