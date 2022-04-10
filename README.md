# quant-korean-market

### tools
- python Backtrader framework
- Creon API

### optimization
- 4주 간 strategy의 parameter에 대한 optimization 수행
- 직후 1주 간 best optimization result에 대한 test 수행
- window를 1주 단위로 이동시키며 optimization 및 test 진행
- method: bayesian or gridsearch

### strategies
- HitAndRun: 
  - volume 및 high가 일정 기준 이상 증가 시 매수, volume 및 high가 일정 기준 이상 하락 시 매도
  - parameters: 매수기준 volume 증가폭 및 high 증가폭, 매도기준 volume 하락폭 및 high 하락폭
- ARIMA: TBU
- OutsidePredict: TBU
- UnusualPrice: TBU
