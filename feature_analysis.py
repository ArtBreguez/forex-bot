import pandas as pd
import numpy as np
import time

def analyze_features():
    path = r"c:\Users\Aline Fernanda\Downloads\project\ticker_eurusd_data.parquet"
    print(f"Lendo dados para pesquisa quantitativa...")
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Resample 15min
    df = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    print(f"Total de amostras (15M): {len(df)}")
    
    # Target: Retorno do PRÓXIMO candle (o que queremos prever)
    df['target_return'] = df['close'].pct_change().shift(-1)
    
    # --- Engenharia de Features ---
    # 1. Momentum / Tendência
    df['sma20'] = df['close'].rolling(20).mean()
    df['dist_sma20'] = (df['close'] - df['sma20']) / df['sma20']
    df['slope_sma20'] = df['sma20'].pct_change(3)
    
    # 2. RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volatilidade (Standard Deviation e Range)
    df['volatilidade'] = df['close'].rolling(20).std() / df['close']
    df['bar_range'] = (df['high'] - df['low']) / df['close']
    
    # 4. Horários (Filtro de Sessão)
    df['hour'] = df.index.hour
    
    # 5. Lagged Returns (Autocorrelação)
    df['lag_ret1'] = df['close'].pct_change(1)
    df['lag_ret2'] = df['close'].pct_change(2)
    
    features = [
        'dist_sma20', 'slope_sma20', 'rsi', 'volatilidade', 'bar_range', 
        'lag_ret1', 'lag_ret2'
    ]
    
    print("\n" + "="*50)
    print("CORRELAÇÃO (IC) COM O PRÓXIMO RETORNO (15 MINUTOS)")
    print("="*50)
    
    results = []
    for f in features:
        corr = df[f].corr(df['target_return'])
        results.append({'feature': f, 'ic': corr})
        
    results_df = pd.DataFrame(results).sort_values(by='ic', ascending=False)
    print(results_df)

    print("\nANÁLISE POR SESSÃO (HORA DO DIA)")
    # Vamos ver se o RSI funciona melhor em certas horas
    for h in [3, 8, 13, 20]: # Tokyo, London, NY, Asia Close
        session_df = df[df['hour'] == h]
        rsi_corr = session_df['rsi'].corr(session_df['target_return'])
        print(f"Hora {h:02d}h UTC: Correlação RSI vs Next Return: {rsi_corr:.4f}")

    print("\nANÁLISE DE VOLATILIDADE")
    print(f"Média Volatilidade: {df['volatilidade'].mean():.6f}")
    print(f"Correlação Volatilidade vs Magnitude do Retorno (Abs): {df['volatilidade'].corr(df['target_return'].abs()):.4f}")

if __name__ == "__main__":
    analyze_features()
