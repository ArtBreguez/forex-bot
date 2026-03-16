import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from falcon_momentum import FalconQuantPremiumReversion

class MT5FalconBridge:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, lot_size=0.1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot_size = lot_size
        self.magic_number = 20260315
        
    def connect(self):
        if not mt5.initialize():
            print("Falha ao inicializar MT5, Erro:", mt5.last_error())
            return False
        print(f"Conectado ao MT5. Terminal: {mt5.terminal_info().name}")
        return True

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

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if signal_type == 1 else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "magic": self.magic_number,
            "comment": "Falcon Premium 15M",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"ERRO MT5: {result.comment} (Retcode: {result.retcode})")
        else:
            print(f">>> ORDEM ENVIADA: {result.order} | Lotes: {lot} | SL: {sl:.5f} | TP: {tp:.5f}")
        return result

    def start_live_loop(self, symbols=["EURUSD", "GBPUSD"]):
        print(f"\n[FALCON LIVE] Iniciando Portfolio Multi-Par: {symbols}")
        print("Monitorando fechamentos de 15 minutos...")
        
        last_processed_bars = {s: None for s in symbols}
        
        while True:
            try:
                for symbol in symbols:
                    self.symbol = symbol # Atualiza o símbolo atual para as funções internas
                    df = self.get_realtime_data(count=600)
                    
                    if df is not None:
                        # Usamos a barra anterior (-2) pois a última (-1) ainda está aberta
                        current_bar_time = df.index[-2]
                        
                        if last_processed_bars[symbol] != current_bar_time:
                            print(f"\n--- [{symbol}] Nova Barra: {current_bar_time} ---")
                            
                            # Rodar Modelo
                            model = FalconQuantPremiumReversion(df)
                            model.apply_trading_logic()
                            
                            signal_row = model.data.iloc[-2]
                            
                            if signal_row['buy_signal'] == 1:
                                curr_p = mt5.symbol_info_tick(symbol).ask
                                self.execute_order(1, curr_p, signal_row['atr'])
                            elif signal_row['sell_signal'] == 1:
                                curr_p = mt5.symbol_info_tick(symbol).bid
                                self.execute_order(-1, curr_p, signal_row['atr'])
                            else:
                                print(f"[{symbol}] Aguardando... Score L: {signal_row['score_long']:.2f} | S: {signal_row['score_short']:.2f}")
                            
                            last_processed_bars[symbol] = current_bar_time
                
                time.sleep(10) # Verifica o ciclo a cada 10 segundos
                
            except KeyboardInterrupt:
                mt5.shutdown()
                break
            except Exception as e:
                print(f"Erro Crítico: {e}")
                time.sleep(30)

if __name__ == "__main__":
    bridge = MT5FalconBridge()
    if bridge.connect():
        bridge.start_live_loop(symbols=["EURUSD", "GBPUSD"])
