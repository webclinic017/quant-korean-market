from itertools import product
import backtrader as bt
from skopt import gp_minimize
from skopt.utils import use_named_args


def backtest(df, strategy, cash=1e6, comm=0.00015, strat_args=None, strat_kwargs=None):
    """method for backtest"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(comm)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    if not strat_args and not strat_kwargs:
        cerebro.addstrategy(strategy)
    elif strat_args and not strat_kwargs:
        cerebro.addstrategy(strategy, *strat_args)
    elif not strat_args and strat_kwargs:
        cerebro.addstrategy(strategy, **strat_kwargs)
    else:
        cerebro.addstrategy(strategy, *strat_args, **strat_kwargs)
    cerebro.run()

    return cerebro


def hyp_grid(df, strategy, space, cash=1e6, comm=0.00015, default_params=None):
    """hyperparameter tuning of strategy with gridsearch method"""
    params_result = []
    best_revenue = 0

    for i, values in enumerate(product(*space.values())):
        params = dict(zip(space.keys(), values))
        if default_params:
            kwargs = dict(params, **default_params)
        else:
            kwargs = params
        bt_output = backtest(df, strategy, cash, comm, strat_kwargs=kwargs)
        strategy_revenue = bt_output.broker.fundvalue / 100
        params_result.append((strategy_revenue, params))

        if best_revenue < strategy_revenue:
            best_revenue = strategy_revenue
        print(f'iteration #{i+1} | obtained: {strategy_revenue:.4f} | current_best: {best_revenue:.4f}')
        print(f'params: {params}')
        print('---------------------------------------------')

    return params_result


def hyp_bayesian(df, strategy, space, cash=1e6, comm=0.00015, n_iters=10, n_initial_points=2, default_params=None):
    """hyperparameter tuning of strategy with bayesian method"""
    @use_named_args(space)
    def obj(**params):
        if default_params:
            kwargs = dict(params, **default_params)
        else:
            kwargs = params
        bt_output = backtest(df, strategy, cash, comm, strat_kwargs=kwargs)
        strategy_revenue = bt_output.broker.fundvalue / 100
        return -strategy_revenue  # 수익률 maximize 위해 부호 반대로 하여 return

    result = gp_minimize(obj, space, n_calls=n_iters, n_initial_points=n_initial_points, verbose=True, random_state=42)

    params_result = [
        (-result.func_vals[iter_],
         {hyp_name: result.x_iters[iter_][hyp_idx] for hyp_idx, hyp_name in enumerate(result.space.dimension_names)})
        for iter_ in range(len(result.x_iters))
    ]
    return params_result
