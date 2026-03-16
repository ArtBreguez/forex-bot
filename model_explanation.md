# Documentação Técnica: Estratégia LightGBM + CVaR

Esta estratégia é baseada no artigo "A Hybrid Model for Financial Time Series Forecasting" (Chen et al. 2020) e adaptada para o par EURUSD no timeframe de 15 minutos (15M).

## 1. Arquitetura do Modelo
O modelo utiliza uma abordagem híbrida que combina **Feature Engineering** avançada com um motor de **Scoring Ponderado** e filtros de **Risco Estocástico**.

### A. Motor de Scoring (10 Fatores)
Cada fator técnico é normalizado via função Sigmóide e ponderado conforme sua importância estatística:

| Fator | Peso | Descrição |
| :--- | :--- | :--- |
| **Momentum** | 235 | Z-Score do preço em relação à média móvel de 50 períodos. |
| **ROC** | 42 | Rate of Change (10 períodos). Mede a velocidade do preço. |
| **ADX/Trend** | 18 | Filtro de tendência para garantir que o mercado não está em lateralidade. |
| **MACD** | 15 | Convergência e divergência de médias para confirmação de impulso. |
| **RSI** | 12 | Relative Strength Index com scoring piecewise (sobrecompra/sobrevenda). |
| **Bands** | 10 | Bollinger Bands para medir a expansão de volatilidade. |
| **Stochastic** | 8 | Oscilador para encontrar exaustão em níveis extremos. |
| **CCI** | 6 | Commodity Channel Index para identificar novos ciclos. |
| **Rel. ATR** | 4 | Razão do ATR atual/médio para evitar mercados "mortos". |
| **Rel. Vol** | 2 | Confirma que o movimento tem volume institucional por trás. |

**Regra de Entrada**: 
- `Score Total >= 0.58`
- Pelo menos **5 indicadores** devem concordar com a direção.

## 2. Gestão de Risco Protetiva
A estratégia foca na sobrevivência do capital a longo prazo.

- **Parametric CVaR (95%)**: Calcula a perda esperada no pior cenário de cauda. Se o risco ultrapassar 2.5%, o trade é bloqueado.
- **Risco Fixo**: Cada operação arrisca exatamente **$5.00** do capital inicial.
- **Slippage & Swap**: O backtest inclui custos reais de latência e taxas noturnas.

## 3. Execução "Fort Knox" (Realismo Total)
Para evitar o erro comum de olhar para o futuro (look-ahead bias):
- **Next-Bar Order**: O sinal é processado no fechamento da Barra 1, mas a execução só ocorre no **Open da Barra 2**.
- **Stepped Trailing**: O Stop Loss é movido em degraus (1.2 -> 0.8 -> 0.5 ATR) conforme o lucro cresce, protegendo o lucro parcial.
- **Partial Close**: Realização de 50% da posição em 2x ATR.

## 4. Próximos Passos
1.  **Dynamic Weighting**: Implementar um algoritmo que ajusta os pesos (235, 42, etc) automaticamente com base na performance recente.
2.  **Portfolio Multi-Pair**: Expandir para GBPUSD e USDJPY para diluir o risco específico do Euro.
3.  **News Filter Integration**: API para bloquear trades durante anúncios de taxas do FED ou BCE.
