import pandas as pd
import numpy as np
import time

def analyze_premium_features():
    path = r"c:\Users\Aline Fernanda\Downloads\project\ticker_eurusd_data.parquet"
    print(f"Lendo dados para pesquisa quantitativa PREMIUM...")
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Precisamos de Volume para VWAP
    df_15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    print(f"Total de amostras (15M): {len(df_15)}")
    
    # Target: Retorno do PRÓXIMO candle
    df_15['target_return'] = df_15['close'].pct_change().shift(-1)
    
    # --- Engenharia de Features PREMIUM ---
    
    # 1. VWAP (Volume Weighted Average Price) - Desvio do Preço
    # Calculando VWAP Diário (Ancorado)
    df_15['date'] = df_15.index.date
    vwap_num = (df_15['close'] * df_15['volume']).groupby(df_15['date']).cumsum()
    vwap_den = df_15['volume'].groupby(df_15['date']).cumsum()
    df_15['vwap'] = vwap_num / vwap_den
    df_15['dist_vwap'] = (df_15['close'] - df_15['vwap']) / df_15['vwap']

    # 2. Donchian Channels (Momentum de Rompimento vs Reversão)
    df_15['dc_upper'] = df_15['high'].rolling(20).max()
    df_15['dc_lower'] = df_15['low'].rolling(20).min()
    df_15['dc_pos'] = (df_15['close'] - df_15['dc_lower']) / (df_15['dc_upper'] - df_15['dc_lower'])

    # 3. Hull Moving Average (HMA) - Suavização Robusta
    def hma(series, n):
        wma_half = series.rolling(n // 2).apply(lambda x: np.dot(x, np.arange(1, n // 2 + 1)) / (n // 2 * (n // 2 + 1) / 2), raw=True)
        wma_full = series.rolling(n).apply(lambda x: np.dot(x, np.arange(1, n + 1)) / (n * (n + 1) / 2), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(int(np.sqrt(n))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(n)) + 1)) / (int(np.sqrt(n)) * (int(np.sqrt(n)) + 1) / 2), raw=True)
    
    print("Calculando HMA (Pode demorar um pouco)...")
    df_15['hma_20'] = hma(df_15['close'], 20)
    df_15['slope_hma'] = df_15['hma_20'].diff() / df_15['close']

    # 4. Volatilidade Dinâmica (Z-Score do Vol)
    df_15['vol_ma'] = df_15['volume'].rolling(20).mean()
    df_15['vol_std'] = df_15['volume'].rolling(20).std()
    df_15['vol_zscore'] = (df_15['volume'] - df_15['vol_ma']) / df_15['vol_std']

    features = ['dist_vwap', 'dc_pos', 'slope_hma', 'vol_zscore']
    
    print("\n" + "="*50)
    print("RANKING FEATURES PREMIUM (IC)")
    print("="*50)
    
    results = []
    for f in features:
        corr = df_15[f].corr(df_15['target_return'])
        results.append({'feature': f, 'ic': corr})
        
    results_df = pd.DataFrame(results).sort_values(by='ic', ascending=False)
    print(results_df)

    # Análise de "Volatilidade Ativa"
    # RSI funciona quando Volatilidade é baixa? Ou quando Z-Score do Volume é alto?
    df_15['rsi'] = 100 - (100 / (1 + (df_15['close'].diff().where(df_15['close'].diff() > 0, 0).rolling(14).mean() / 
                                    (-df_15['close'].diff().where(df_15['close'].diff() < 0, 0).rolling(14).mean()))))
    
    high_vol_session = df_15[df_15['vol_zscore'] > 2]
    low_vol_session = df_15[df_15['vol_zscore'] < -0.5]
    
    print("\nANÁLISE DE CONTEXTO (VOLUMETRIA)")
    print(f"IC do RSI em ALTO VOLUME: {high_vol_session['rsi'].corr(high_vol_session['target_return']):.4f}")
    print(f"IC do RSI em BAIXO VOLUME: {low_vol_session['rsi'].corr(low_vol_session['target_return']):.4f}")

if __name__ == "__main__":
    analyze_premium_features()
