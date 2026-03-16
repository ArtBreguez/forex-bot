import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime
from falcon_elite_alpha import FalconEliteAlpha

# Configuração de Logging Institucional
log_filename = "falcon_live_elite.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FalconElite")

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
        
        # Mapeamento de Bitmask SYMBOL_FILLING: bit 1=FOK, bit 2=IOC
        filling_mode = symbol_info.filling_mode
        if filling_mode & 1: 
            return mt5.ORDER_FILLING_FOK # 0
        elif filling_mode & 2: 
            return mt5.ORDER_FILLING_IOC # 1
        else:
            return mt5.ORDER_FILLING_RETURN # 2

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
            "comment": "FEA-Initial", # Falcon Elite Alpha
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[{self.symbol}] ERRO MT5: {result.comment} (Retcode: {result.retcode})")
        else:
            logger.info(f"[{self.symbol}] [OK] ORDEM EXECUTADA: {result.order} | Lotes: {lot} | SL: {sl:.5f} | TP: {tp:.5f}")
        return result

    def manage_open_positions(self, atr_value):
        """
        Gerencia Trailing Stop e Saída Parcial para posições abertas.
        """
        positions = mt5.positions_get(magic=self.magic_number)
        if not positions: return

        for pos in positions:
            if pos.symbol != self.symbol: continue

            ticket = pos.ticket
            entry_price = pos.price_open
            current_price = pos.price_current
            sl_current = pos.sl
            tp_current = pos.tp
            volume = pos.volume
            order_type = pos.type # 0 for BUY, 1 for SELL
            
            # Direção do lucro
            side = 1 if order_type == mt5.POSITION_TYPE_BUY else -1
            profit_pips = (current_price - entry_price) * side * 10000
            profit_atr = profit_pips / (atr_value * 10000 + 1e-9)
            
            # 1. Saída Parcial (50%) - se ainda não foi feita (checado pelo comment)
            if profit_atr >= 2.0 and "Partial" not in pos.comment:
                close_vol = round(volume / 2, 2)
                if close_vol >= 0.01:
                    logger.info(f"[{self.symbol}] Trade {ticket}: Realizando Lucro Parcial (2.0x ATR)")
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": close_vol,
                        "type": mt5.ORDER_TYPE_SELL if order_type == 0 else mt5.ORDER_TYPE_BUY,
                        "position": ticket,
                        "price": mt5.symbol_info_tick(self.symbol).bid if order_type == 0 else mt5.symbol_info_tick(self.symbol).ask,
                        "comment": "FEA-Partial",
                        "type_filling": self.get_filling_mode(),
                    }
                    mt5.order_send(request)

            # 2. Stepped Trailing Stop
            new_sl = sl_current
            # Trail distances baseadas no Backtest
            if profit_atr >= 3.0:
                # Trail a 0.5 ATR de distância do preço atual
                new_sl = current_price - (atr_value * 0.5 * side)
            elif profit_atr >= 2.0:
                new_sl = current_price - (atr_value * 0.8 * side)
            elif profit_atr >= 1.0:
                new_sl = current_price - (atr_value * 1.2 * side)
            
            # Garantir que o SL só se move a favor (nunca aumenta o prejuízo)
            if side == 1: # BUY
                if new_sl > sl_current + 0.00001: 
                    self.modify_sl_tp(ticket, new_sl, tp_current)
            else: # SELL
                if new_sl < sl_current - 0.00001 and sl_current != 0:
                    self.modify_sl_tp(ticket, new_sl, tp_current)

    def modify_sl_tp(self, ticket, sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(sl, 5),
            "tp": round(tp, 5)
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{self.symbol}] Trade {ticket}: Stop Loss atualizado para {sl:.5f}")
        else:
            logger.debug(f"Falha ao Modificar SL: {result.comment}")

    def start_live_loop(self, symbols=["EURUSD#", "GBPUSD#"]):
        logger.info(f"Portfolio Monitor Ativado: {symbols} (M15)")
        print("\n" + "="*50)
        print("   🦅 FALCON ELITE LIVE - 24/7 MONITORING   ")
        print("="*50)
        
        last_processed_bars = {s: None for s in symbols}
        
        while True:
            try:
                dashboard_lines = []
                for symbol in symbols:
                    self.symbol = symbol
                    df = self.get_realtime_data(count=600)
                    
                    if df is not None:
                        # Rodar Modelo para obter scores atuais (Live e Closed)
                        model = FalconEliteAlpha(df)
                        model.apply_trading_logic()
                        
                        live_row = model.data.iloc[-1]
                        current_bar_time = model.data.index[-2]
                        
                        # Dashboard Info
                        score_l = live_row['score_long']
                        score_s = live_row['score_short']
                        tick_time = datetime.now().strftime('%H:%M:%S')
                        
                        line = f"[{symbol}] {tick_time} | Bar {model.data.index[-1].strftime('%H:%M')} | Score L: {score_l:.2f} S: {score_s:.2f}"
                        
                        # Monitoramento de Posições Abertas
                        self.manage_open_positions(live_row['atr'])
                        open_pos = mt5.positions_get(symbol=symbol, magic=self.magic_number)
                        if open_pos:
                            p = open_pos[0]
                            side = "BUY" if p.type == 0 else "SELL"
                            p_atr = ((p.price_current - p.price_open) * (1 if p.type==0 else -1) * 10000) / (live_row['atr'] * 10000 + 1e-9)
                            line += f" | {side} {p.volume} | Profit: {p_atr:.1f} ATR"
                        
                        dashboard_lines.append(line)

                        if last_processed_bars[symbol] != current_bar_time:
                            # Detectada nova barra fechada -> Processar Sinal de Execução
                            signal_row = model.data.iloc[-2]
                            
                            if signal_row['buy_signal'] == 1:
                                logger.info(f"[{symbol}] [BUY] SINAL DE COMPRA CONFIRMADO (Barra {current_bar_time})")
                                curr_p = mt5.symbol_info_tick(symbol).ask
                                self.execute_order(1, curr_p, signal_row['atr'])
                            elif signal_row['sell_signal'] == 1:
                                logger.info(f"[{symbol}] [SELL] SINAL DE VENDA CONFIRMADO (Barra {current_bar_time})")
                                curr_p = mt5.symbol_info_tick(symbol).bid
                                self.execute_order(-1, curr_p, signal_row['atr'])
                            
                            last_processed_bars[symbol] = current_bar_time
                
                # Imprimir Dashboard Multi-Lote
                # Limpa as N linhas anteriores aproximadas com \r e \033[F (ANSI move up)
                print("\033[H\033[J", end="") # Limpa tela ANSI (funciona no Windows Terminal)
                print("="*50)
                print("   🦅 FALCON ELITE LIVE - 24/7 MONITORING   ")
                print("="*50)
                for line in dashboard_lines:
                    print(line)
                print("="*50)
                
                time.sleep(15) 
                
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
        bridge.start_live_loop(symbols=["EURUSD#", "GBPUSD#"])
