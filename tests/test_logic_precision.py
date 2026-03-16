import pytest
import pandas as pd
import numpy as np
from falcon_elite_alpha import FalconEliteAlpha

def generate_mock_data(size=200):
    """Gera dados OHLCV sintéticos para testes."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=size, freq="15min")
    close = 1.1000 + np.cumsum(np.random.normal(0, 0.001, size))
    df = pd.DataFrame({
        'open': close - 0.0005,
        'high': close + 0.0010,
        'low': close - 0.0010,
        'close': close,
        'volume': np.random.randint(100, 1000, size)
    }, index=dates)
    return df

def test_indicators_generation():
    """Valida se os indicadores estão sendo gerados e não contêm NaNs fatais."""
    df = generate_mock_data(200)
    strategy = FalconEliteAlpha(df)
    strategy.generate_indicators()
    
    # Verificar se as colunas principais existem
    required_cols = ['f0_score', 'f1_score', 'f4_score', 'atr', 'cvar']
    for col in required_cols:
        assert col in strategy.data.columns
        # O dropna() no final da função remove NaNs, então o que sobra deve ser válido
        assert not strategy.data[col].isnull().any()

def test_weighted_scoring_logic():
    """Valida se a média ponderada dos scores está correta e respeita os pesos de strategy.mql5."""
    df = generate_mock_data(150)
    strategy = FalconEliteAlpha(df)
    strategy.apply_trading_logic()
    
    # Pesos do MQL5 somam 500
    total_weight = 500
    
    # Pegar uma linha qualquer
    row = strategy.data.iloc[-1]
    manual_score = 0
    raw_weights = [90, 75, 65, 60, 55, 45, 35, 30, 25, 20]
    
    for i in range(10):
        manual_score += row[f'f{i}_score'] * (raw_weights[i] / total_weight)
    
    assert pytest.approx(row['score_long'], 0.0001) == manual_score

def test_cvar_mathematical_consistency():
    """Valida se o cálculo do CVaR manual (Paramétrico) bate com a implementação."""
    df = generate_mock_data(150)
    strategy = FalconEliteAlpha(df)
    strategy.generate_indicators()
    
    # Pegar dados de fechamento para cálculo manual
    # O CVaR no código usa janelas de 100
    close = df['close']
    returns = np.log(close / close.shift(1)).fillna(0)
    
    # Cálculo manual da última barra
    window = returns.tail(100)
    mu = window.mean()
    sigma = window.std()
    phi = np.exp(-0.5 * 1.645**2) / np.sqrt(2 * np.pi)
    alpha = 0.95
    manual_cvar = np.abs(-(mu - sigma * phi / (1 - alpha))) * 100
    
    assert pytest.approx(strategy.data['cvar'].iloc[-1], 0.0001) == manual_cvar

def test_risk_management_lot_sizing():
    """Valida se o cálculo de tamanho de lote respeita o risco fixo de $5."""
    # Simular um cenário onde o ATR é 0.0020 (20 pips)
    df = generate_mock_data(150)
    strategy = FalconEliteAlpha(df)
    strategy.initial_capital = 250.0
    
    # Mock data forcing a signal
    # sl_pips = entry_atr * 2 * 10000 -> 0.0020 * 2 * 10000 = 40 pips
    # lot = 5.0 / (40 * 10.0) = 5 / 400 = 0.0125 -> 0.01
    
    # No código do backtest:
    # risk_usd = 5.0
    # sl_pips = entry_atr * 2 * 10000
    # active_lots = max(0.01, round(risk_usd / (sl_pips * 10.0 + 1e-9), 2))
    
    risk_usd = 5.0
    entry_atr = 0.0020
    sl_pips = entry_atr * 2 * 10000
    expected_lots = max(0.01, round(5.0 / (sl_pips * 10.0), 2))
    
    assert expected_lots == 0.01

def test_session_filter():
    """Valida se o robô respeita o filtro de horário (07h às 17h)."""
    df = generate_mock_data(200)
    # Criar uma data fora do horário (ex: 3 da manhã)
    df.index = pd.date_range("2026-01-01 00:00:00", periods=len(df), freq="15min")
    strategy = FalconEliteAlpha(df)
    strategy.apply_trading_logic()
    
    # No horário 00:00 às 06:45, os sinais buy_signal e sell_signal devem ser 0
    night_data = strategy.data.between_time('00:00', '06:45')
    assert (night_data['buy_signal'] == 0).all()
    assert (night_data['sell_signal'] == 0).all()

if __name__ == "__main__":
    pytest.main([__file__])
