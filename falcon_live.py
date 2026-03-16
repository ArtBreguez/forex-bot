import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime
from falcon_momentum import FalconQuantPremiumReversion

# Configuração de Logging Institucional
log_filename = "falcon_live.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FalconLive")

class MT5FalconBridge:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, lot_size=0.1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot_size = lot_size
        self.magic_number = 20260315
        
    def connect(self):
        if not mt5.initialize():
            logger.error(f"Falha ao inicializar MT5, Erro: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Conectado ao MT5. Conta: {account_info.login} | Corretora: {account_info.company}")
            logger.info(f"Saldo Atual: ${account_info.balance:.2f} | Alavancagem: 1:{account_info.leverage}")
        return True
    
    def get_filling_mode(self):
        """Detecta automaticamente o tipo de preenchimento suportado pela corretora/par."""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None: return mt5.ORDER_FILLING_IOC
        
        filling_mode = symbol_info.filling_mode
        if filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def get_realtime_data(self, count=500):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        if rates is None:
            print(f"Erro ao copiar rates para {self.symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Ajustar nomes para o padrão da estratégia
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def execute_order(self, signal_type, entry_price, atr_value):
        """
        Calcula lote fixo para arriscar $5 (2% de $250) baseado no SL de 2x ATR.
        """
        mt5.symbol_select(self.symbol, True)
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None: return None

        # 1. Cálculo do Lote (Risco Fixo $5.0)
        sl_dist = atr_value * 2
        sl_pips = sl_dist * 10000
        
        # No MT5 Standard (1 lote = 100k), 1 pip no micro lote (0.01) vale ~$0.10.
        lot = max(0.01, round(5.0 / (sl_pips * 10.0 + 1e-9), 2))
        
        # 2. Proteção de Margem (Limite de 0.03 para $250 na XM)
        lot = min(lot, 0.03)
        
        # 3. Preços de Execução
        point = symbol_info.point
        sl = entry_price - signal_type * sl_dist
        tp = entry_price + signal_type * (atr_value * 5)

        filling = self.get_filling_mode()
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if signal_type == 1 else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "magic": self.magic_number,
            "comment": "Falcon Elite 15M",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[{self.symbol}] ERRO MT5: {result.comment} (Retcode: {result.retcode})")
        else:
            logger.info(f"[{self.symbol}] ✅ ORDEM EXECUTADA: {result.order} | Lotes: {lot} | SL: {sl:.5f} | TP: {tp:.5f}")
        return result

    def start_live_loop(self, symbols=["EURUSD", "GBPUSD"]):
        logger.info(f"Portfolio Monitor Ativado: {symbols} (M15)")
        print("\n" + "="*50)
        print("   🦅 FALCON ELITE LIVE - 24/7 MONITORING   ")
        print("="*50 + "\n")
        
        last_processed_bars = {s: None for s in symbols}
        
        while True:
            try:
                for symbol in symbols:
                    self.symbol = symbol
                    df = self.get_realtime_data(count=600)
                    
                    if df is not None:
                        current_bar_time = df.index[-2]
                        
                        if last_processed_bars[symbol] != current_bar_time:
                            # Rodar Modelo
                            model = FalconQuantPremiumReversion(df)
                            model.apply_trading_logic()
                            
                            signal_row = model.data.iloc[-2]
                            
                            score_l = signal_row['score_long']
                            score_s = signal_row['score_short']
                            
                            status_msg = f"[{symbol}] Bar: {current_bar_time} | L: {score_l:.2f} | S: {score_s:.2f}"
                            
                            if signal_row['buy_signal'] == 1:
                                logger.info(f"[{symbol}] 🟢 SINAL DE COMPRA DETECTADO")
                                curr_p = mt5.symbol_info_tick(symbol).ask
                                self.execute_order(1, curr_p, signal_row['atr'])
                            elif signal_row['sell_signal'] == 1:
                                logger.info(f"[{symbol}] 🔴 SINAL DE VENDA DETECTADO")
                                curr_p = mt5.symbol_info_tick(symbol).bid
                                self.execute_order(-1, curr_p, signal_row['atr'])
                            else:
                                print(f"\r{status_msg}", end="", flush=True)
                            
                            last_processed_bars[symbol] = current_bar_time
                
                time.sleep(20) # Reduzido impacto de pooling
                
            except KeyboardInterrupt:
                logger.info("Desligando robô...")
                mt5.shutdown()
                break
            except Exception as e:
                logger.error(f"Erro no Loop Principal: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bridge = MT5FalconBridge()
    if bridge.connect():
        bridge.start_live_loop(symbols=["EURUSD", "GBPUSD"])
