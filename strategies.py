import os
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import backtrader as bt
from skopt.space import Real, Integer

from data_loader import CpStockChart
from backtest_base import backtest, hyp_grid, hyp_bayesian
import utils

sns.set_style('darkgrid')


class HitAndRun(bt.Strategy):
    """
    Hit-and-Run 전략
    1. 거래시간: 9:10 ~ 15:10
    2. 장중 매수조건: 2분 전에 비해 1분 전 volume이 (hit_volume_rate)배 이상 증가 and
                      2분 전에 비해 1분 전 high가 (hit_high_unit)*거래단위 이상 증가
    3. 장중 매도조건: 2분 전에 비해 1분 전 volume이 (run_volume_rate)배 이하로 감소 or
                      1분 전 high가 매수시점 high보다 (run_high_unit)*거래단위 이상 하락
    4. 장외시간에 보유할지는 선택사항 -> outside_hold
    """
    def __init__(self, code, hit_volume_rate=5, hit_high_unit=0, run_volume_rate=0.2, run_high_unit=0, outside_hold=True):
        self.code = code
        self.hit_volume_rate = hit_volume_rate
        self.hit_high_unit = hit_high_unit
        self.run_volume_rate = run_volume_rate
        self.run_high_unit = run_high_unit
        self.outside_hold = outside_hold

        self.high_long_start = 0  # 매수시점 close


    def __str__(self):
        return 'HitAndRun'


    def next(self):
        data = self.datas[0]
        unit = utils.price_unit(data.high[-1], self.code) # 2분 전 high 기준으로 거래단위 계산

        # 장 시작 10분 후 - 장 마감 10분 전 거래전략
        if 910 <= int(self.datetime.datetime(ago=0).strftime('%H%M')) < 1510:
            if not self.position:
                if (data.volume[0] / data.volume[-1] >= self.hit_volume_rate) and (data.high[0] - data.high[-1] >= self.hit_high_unit * unit):
                    size = self.broker.get_cash() // data.close[0]
                    self.buy(price=data.close[0], size=size)
                    self.high_long_start = data.high[0]
            else:
                if (data.volume[0] / data.volume[-1] <= self.run_volume_rate) or (data.close[0] - self.high_long_start <= -self.run_high_unit * unit):
                    self.close()
                    self.high_long_start = 0

        # 장 마감 10분 전 매도
        elif int(self.datetime.datetime(ago=0).strftime('%H%M')) == 1510:
            if self.position:
                self.close()
                self.high_long_start = 0

        # 장외 시간 보유하기 - 폐장 시 매수
        elif self.outside_hold and int(self.datetime.datetime(ago=0).strftime('%H%M')) == 1520:
            if not self.position:
                size = self.broker.get_cash() // data.close[0]
                self.buy(price=data.close[0], size=size)
                self.high_long_start = data.high[0]

        # 장외 시간 보유하기 - 개장 시 매도
        elif self.outside_hold and int(self.datetime.datetime(ago=0).strftime('%H%M')) == 901:
            if self.position:
                self.close()
                self.high_long_start = 0


class Arima(bt.Strategy):
    """
    TODO: ARIMA 활용 전략
    """
    def __init__(self):
        pass


    def __str__(self):
        return 'Arima'


    def next(self):
        pass


class OutsidePredict(bt.Strategy):
    """
    TODO 장중 데이터로 LSTM 모델 학습시켜 장외 오를지 예측
    배경: 카카오게임즈는 평균적으로 장중보다 장외에 많이 오르는 특성 가짐
    """
    def __init__(self):
        pass


    def __str__(self):
        return 'OutsidePredict'


    def next(self):
        pass


class UnusualPrice(bt.Strategy):
    """
    TODO 특이한 가격대 근처에서 주가 움직임이 다르다는 가설, 이를 투자에 활용 가능한지 확인.
    ex. SK하이닉스 호가가 100,000 근처일 때
    """
    def __init__(self):
        pass


    def __str__(self):
        return 'UnusualPrice'


    def next(self):
        pass


class StrategyTest:
    """
    duration window를 1주일씩 이동시키며 최적화를 진행.
    opt duration: 4주 / test duration: 1주
    """
    def __init__(self, strategy, name_list, analyze_from_date, analyze_to_date, outside_hold, opt_method, search_space,
                 fig_save=True, fig_show=True):
        self.strategy = strategy
        self.name_list = name_list
        self.analyze_from_date = analyze_from_date
        self.analyze_to_date = analyze_to_date
        self.outside_hold = outside_hold

        assert opt_method in ('gridsearch', 'bayesian'), "variable 'opt_method' must be in ('gridsearch', 'bayesian')."
        self.opt_method = opt_method

        if opt_method == 'gridsearch':
            assert type(search_space) == dict, \
                "Variable 'search_space' must be a dictionary. Example is below:\n" \
                "{'hit_volume_rate': 10 ** (np.arange(0, 1.1, 0.1)), \n" \
                " 'hit_high_unit': np.arange(0, 11, 1)}"
        elif opt_method == 'bayesian':
            assert type(search_space) == list, \
                "Variable 'search_space' must be a list. Example is below:\n" \
                "[Real(1, 10, 'log-uniform', name='hit_volume_rate'), \n" \
                " Integer(0, 10, name='hit_high_unit')]"
        self.search_space = search_space

        self.fig_save = fig_save
        self.fig_show = fig_show

        self.result, self.tracking = self._test()


    def tracking_plot(self, save=True, show=True):
        for name in self.name_list:
            plt.figure()
            plt.plot([x[0] for x in self.tracking[name]], c=sns.color_palette('Paired')[1])
            plt.plot([x[1] for x in self.tracking[name]], c=sns.color_palette('Paired')[7])
            plt.plot([x[2] for x in self.tracking[name]], c=sns.color_palette('Paired')[0], linestyle='--')
            plt.plot([x[3] for x in self.tracking[name]], c=sns.color_palette('Paired')[6], linestyle='--')

            tracking_len = len(self.tracking[name])
            tick_list = [0, (tracking_len - 1) // 2, tracking_len - 1]
            plt.xticks(ticks=tick_list,
                       labels=[f'opt: {self.result[name][t][0]} ~ {self.result[name][t][1]}\n'
                               f'test: {self.result[name][t][2]} ~ {self.result[name][t][3]}'
                               for t in tick_list],
                       size=8)
            plt.ylabel('rate_of_return')
            plt.legend(['best_opt_opt_RoR', 'best_opt_test_RoR', 'opt_baseline_RoR', 'test_baseline_RoR'])
            plt.title(name, fontdict={'family': 'Malgun Gothic'})

            if save:
                plt.savefig(f'plot/{self.strategy.__name__}_{name}.png')
            if show:
                plt.show()


    def _test(self):
        """종목 리스트, 분석 구간에 대해 test 수행"""
        total_result = {name: [] for name in self.name_list}  # 전체 결과 저장
        total_tracking = {name: [] for name in self.name_list}  # 각 duration에 대한 opt best hyp의 opt, test 결과를 baseline과 함께 저장

        for name in self.name_list:
            with open('./data/stock_code_name.pickle', 'rb') as fr:
                code_name_dict = pickle.load(fr)
                name_code_dict = pickle.load(fr)
            code = name_code_dict[name]

            durations = [(dt.strftime('%Y%m%d'),
                          (dt + timedelta(days=25)).strftime('%Y%m%d'),
                          (dt + timedelta(days=28)).strftime('%Y%m%d'),
                          (dt + timedelta(days=32)).strftime('%Y%m%d'))
                         for dt in np.arange(self.analyze_from_date, self.analyze_to_date, timedelta(days=1), dtype=datetime)
                         if dt.isoweekday() == 1 and dt + timedelta(days=32) <= self.analyze_to_date]

            for opt_from_date, opt_to_date, test_from_date, test_to_date in durations:
                print('\033[93m' +  # yellow
                      f'name: {name} | duration: opt {opt_from_date}~{opt_to_date}, test {test_from_date}~{test_to_date}' +
                      '\033[0m')  # reset

                # 주가데이터 로드
                objStockChart = CpStockChart()
                df_opt = objStockChart.request_minute(code=code, minute_range=1, from_date=opt_from_date, to_date=opt_to_date)
                df_opt = utils.set_datetime_as_index(df_opt, drop=False)
                df_opt_daily = objStockChart.request_dwm(code=code, dwm='D', from_date=opt_from_date, to_date=opt_to_date)

                df_test = objStockChart.request_minute(code=code, minute_range=1, from_date=test_from_date, to_date=test_to_date)
                df_test = utils.set_datetime_as_index(df_test, drop=False)
                df_test_daily = objStockChart.request_dwm(code=code, dwm='D', from_date=test_from_date, to_date=test_to_date)
                print('data load is complete.')

                if self.outside_hold:
                    opt_baseline_ror = df_opt.iloc[-1].close / df_opt.iloc[0].open
                    test_baseline_ror = df_test.iloc[-1].close / df_test.iloc[0].open
                else:
                    opt_baseline_ror = (df_opt_daily.close / df_opt_daily.open).prod()
                    test_baseline_ror = (df_test_daily.close / df_test_daily.open).prod()

                # optimiaztion
                default_params = {'code': code, 'outside_hold': self.outside_hold}
                if self.opt_method == 'gridsearch':
                    opt_result = hyp_grid(df_opt, self.strategy, self.search_space, cash=1e6, comm=0.00015, default_params=default_params)
                elif self.opt_method == 'bayesian':
                    opt_result = hyp_bayesian(df_opt, self.strategy, self.search_space, cash=1e6, comm=0.00015,
                                              n_iters=50, n_initial_points=10, default_params=default_params)
                else:
                    opt_result = None
                print('optimization is complete.')

                # opt 기간에서의 수익률과 test 기간에서의 수익률 비교
                opt_test_compare = []
                for opt_ror, hyp_set in tqdm(opt_result):
                    bt_output = backtest(df_test, self.strategy, 1e6, 0.00015, strat_kwargs=dict(hyp_set, **default_params))
                    test_ror = bt_output.broker.fundvalue / 100
                    opt_test_compare.append(
                        (opt_ror, test_ror, hyp_set)
                    )
                print('comparing test vs opt is complete.')

                total_result[name].append(
                    (opt_from_date, opt_to_date, test_from_date, test_to_date, opt_test_compare)
                )

                self._plot_test_vs_opt(name, opt_from_date, opt_to_date, test_from_date, test_to_date,
                                      opt_test_compare, opt_baseline_ror, test_baseline_ror,
                                      fig_save=self.fig_save, fig_show=self.fig_show)

                # opt best인 opt, test ror logging
                best_opt_idx = np.argmax(np.array(opt_test_compare)[:, 0])
                best_opt_opt_ror, best_opt_test_ror, _ = opt_test_compare[best_opt_idx]
                total_tracking[name].append(
                    (best_opt_opt_ror, best_opt_test_ror, opt_baseline_ror, test_baseline_ror)
                )

        return total_result, total_tracking


    def _plot_test_vs_opt(self, name, opt_from_date, opt_to_date, test_from_date, test_to_date,
                         opt_test_compare, opt_baseline_ror, test_baseline_ror, fig_save, fig_show):
        """test 기간에서의 수익률 vs opt 기간에서의 수익률 plot"""
        fig, ax = plt.subplots()

        best_opt_idx = np.argmax(np.array(opt_test_compare)[:, 0])
        if self.opt_method == 'gridsearch':
            sns.scatterplot(
                x=[x[0] for x in opt_test_compare], y=[x[1] for x in opt_test_compare],
                s=[100 if i == best_opt_idx else 20 for i in range(len(opt_test_compare))]
            )
        elif self.opt_method == 'bayesian':
            sns.scatterplot(
                x=[x[0] for x in opt_test_compare], y=[x[1] for x in opt_test_compare],
                s=[100 if i == best_opt_idx else 20 for i in range(len(opt_test_compare))],
                c=[i for i in range(len(opt_test_compare))],
                cmap=mpl.colors.LinearSegmentedColormap.from_list('', [[0, sns.color_palette('Paired')[0]], [1, sns.color_palette('Paired')[1]]])
            )

        ax.axvline(opt_baseline_ror, linestyle='--', c='gray')
        ax.axhline(test_baseline_ror, linestyle='--', c='gray')
        ax.set_xlabel('opt_RoR')
        ax.set_ylabel('test_RoR')
        ax.set_title(f'test vs opt | {name} \n '
                     f'test: {test_from_date} ~ {test_to_date} | opt: {opt_from_date} ~ {opt_to_date}',
                     size=12, fontdict={'family': 'Malgun Gothic'})

        if fig_save:
            dir = f'plot/log/{self.strategy.__name__}_{name}'
            if not os.path.exists(dir):
                os.mkdir(dir)
            fig.savefig(dir + f'/test_{test_from_date}_{test_to_date}_opt_{opt_from_date}_{opt_to_date}.png')
        if fig_show:
            fig.show()


if __name__ == '__main__':
    # TODO: 논리적으로 종목 선정
    # TODO: robustness metric 고안

    strategy = HitAndRun
    name_list = ['에코프로비엠', '셀트리온헬스케어', '펄어비스']
    analyze_from_date = datetime.strptime('20211004', '%Y%m%d')
    analyze_to_date = datetime.strptime('20220401', '%Y%m%d')
    outside_hold = True
    opt_method = 'bayesian'
    search_space = [Real(1, 10, 'log-uniform', name='hit_volume_rate'),
                    Integer(0, 10, name='hit_high_unit'),
                    Real(0.1, 1, 'log-uniform', name='run_volume_rate'),
                    Integer(0, 10, name='run_high_unit')]

    test = StrategyTest(strategy, name_list, analyze_from_date, analyze_to_date, outside_hold, opt_method, search_space,
                        fig_save=False, fig_show=True)
    test.tracking_plot()
