import functools
import tempfile
from contextlib import suppress
import datetime
from pathlib import Path

import multiprocessing

import rapidjson
from tqdm.notebook import tqdm
import json
from freqtrade.configuration import TimeRange
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports import generate_backtest_stats, generate_wins_draws_losses, show_backtest_results
from tabulate import tabulate
import quantstats as qs

import pandas as pd  # noqa
from IPython import get_ipython
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
import plotly.graph_objects as go

from freqtrade.configuration import Configuration, TimeRange, validate_config_consistency
from freqtrade.data.btanalysis import load_backtest_data, load_trades_from_db
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_pair_history
from freqtrade.enums import RunMode, CandleType
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import deep_merge_dicts
from freqtrade.plot.plotting import generate_candlestick_graph
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy

"""
freqtrade_analysis_notebook helper functions

Original code from @rk
Contributions from @froggleston

"""


def setup():
    pd.options.display.width = 5000
    pd.options.display.max_colwidth = 5000
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("Notebook setup done")


def load_trades(config: dict, pair: str = None):
    backtest_dir = config['user_data_dir'] / 'backtest_results'
    bt_trades = run_trades = None
    with suppress(ValueError):
        bt_trades = load_backtest_data(backtest_dir, strategy=config['strategy'])
    with suppress(ValueError):
        if 'db_url' in config:
            run_trades = load_trades_from_db(config['db_url'], strategy=config['strategy'])

    for trades in (bt_trades, run_trades):
        if trades is None:
            continue

    return bt_trades, run_trades


def load_dataframe(strategy: IStrategy, pair: str, timerange: str):
    config = strategy.config
    data_location = Path(config['user_data_dir'], 'data', config['exchange']['name'])
    candles = load_pair_history(datadir=data_location,
                                timerange=TimeRange.parse_timerange(timerange),
                                timeframe=config['timeframe'],
                                pair=pair, data_format=config['dataformat_ohlcv'])
    df = strategy.analyze_ticker(candles, {'pair': pair})
    return df, candles


def load_strategy(timeframe=None, config_files=None, config_extra=None, runmode=RunMode.OTHER):
    strategy = None
    config_files = config_files.copy()
    if config_files is None:
        config_files = ['config.json']
    tmp_fp = None
    try:
        if config_extra:
            tmp_fp = tempfile.NamedTemporaryFile('w+')
            json.dump(config_extra, tmp_fp)
            tmp_fp.flush()
            config_files.append(tmp_fp.name)

        conf_object = Configuration({'config': config_files}, runmode)
        config = conf_object.get_config()
        validate_config_consistency(config)
        if config.get('strategy'):
            strategy = StrategyResolver.load_strategy(config)
            config['timeframe'] = timeframe or strategy.timeframe
            strategy.dp = DataProvider(config, None)
    finally:
        if tmp_fp:
            tmp_fp.close()

    return config, strategy


def format_timerange(df, dt):
    return f'{df.year}{df.month:02d}{df.day:02d}-{dt.year}{dt.month:02d}{dt.day:02d}'


def filter_trades(trades: DataFrame, sell_reasons):
    if not sell_reasons or trades is None or trades.empty:
        return trades

    conditions = []
    for sell_reason in sell_reasons:
        conditions.append(trades['sell_reason'] == sell_reason)
    return trades.loc[functools.reduce(lambda a, b: a | b, conditions)].copy()


def trades_freq2quant(series: pd.DataFrame, config: dict):
    """Convert to a format accepted by quantstats."""

    if len(series) > 0:
        daily_profit = series.resample('1d', on='close_date')['profit_abs'].sum().astype(float).round(5)
        daily_profit = daily_profit.rename_axis("Date")

        # Convert to date without timezone
        t = daily_profit.axes[0]
        t = t.tz_convert(None)
        daily_profit = daily_profit.set_axis(t)

        # Generate daily wallet value
        value = config['dry_run_wallet']
        for index, profit in daily_profit.items():
            value += profit
            daily_profit.at[index] = value

        return daily_profit.pct_change()
    else:
        return 0


def split_timerange(timerange):
    # borrowed from https://stackoverflow.com/a/13565185
    # as noted there, the calendar module has a function of its own
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
        return next_month - datetime.timedelta(days=next_month.day)

    timerange = TimeRange.parse_timerange(timerange)
    start = datetime.datetime.utcfromtimestamp(timerange.startts)
    end = datetime.datetime.utcfromtimestamp(timerange.stopts)

    def monthlist(start,end):
        result = []
        while True:
            if start.month == 12:
                next_month = start.replace(year=start.year+1,month=1, day=1)
            else:
                next_month = start.replace(month=start.month+1, day=1)
            if next_month > end:
                break
            result.append(start.strftime("%Y%m%d")+'-'+last_day_of_month(start).strftime("%Y%m%d"))
            start = next_month

        if start != end:
            # don't add timeranges that are 1 day
            result.append(start.strftime("%Y%m%d")+'-'+end.strftime("%Y%m%d"))

        return result

    ml = monthlist(start,end)

    return ml

# Execute backtest jobs
def backtest_one(config, config_i, pairlist_fmt, pair_count, timerange):
    
    config['timerange'] = timerange
    timerange = TimeRange.parse_timerange(timerange)
    bt_date = datetime.datetime.utcfromtimestamp(timerange.startts)

    if pairlist_fmt:
        pairlist_file = pairlist_fmt.format(exchange=config["exchange"]["name"], stake_currency=config["stake_currency"],
                                            pair_count=pair_count, year=bt_date.year, month=bt_date.month, day=bt_date.day)
        with open(pairlist_file) as fp:
            deep_merge_dicts(rapidjson.load(fp, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS), config)
    config['pairs'] = config['exchange']['pair_whitelist']
    
    backtesting = Backtesting(config)
    data, timerange = backtesting.load_bt_data()
    
    backtesting.load_bt_data_detail()
    min_date = max_date = None
    processed_dfs = {}

    for strat in backtesting.strategylist:
        min_date, max_date = backtesting.backtest_one_strategy(strat, data, timerange)
        processed_dfs[strat.get_strategy_name()] = backtesting.processed_dfs[strat.get_strategy_name()]

    comparison_stats = []
    raw_trades = {}
    if len(backtesting.strategylist) > 0:
        stats = generate_backtest_stats(data, backtesting.all_results, min_date=min_date, max_date=max_date)
        show_backtest_results(config, stats)

        comparison_stats = stats['strategy_comparison']
        for i in range(len(comparison_stats)):
            r = comparison_stats[i]
            r['date'] = f'{bt_date.year}-{bt_date.month:02d}-{bt_date.day:02d}'
            r['pair_count'] = pair_count
            r['wdl'] = generate_wins_draws_losses(r['wins'], r['draws'], r['losses'])
            trades_df = backtesting.all_results[r['key']]['results']
            raw_trades[r['key']] = trades_df
    return pair_count, comparison_stats, raw_trades, config, config_i, processed_dfs


def prepare_configs(test_config, timeframe_detail=None, data_location=None, data_format="json", trading_mode=CandleType.SPOT):
    configs = []

    for c in test_config:
        config_files = [cc for cc in c['config'] if isinstance(cc, str)]
        config_extra = {
            'strategy': c['strategy'],
        }

        if config_files is None:
            config_files = ['config.json']

        for cc in c['config']:
            if isinstance(cc, dict):
                config_extra = deep_merge_dicts(cc, config_extra)
        
        tmp_fp = None
        try:
            if config_extra:
                tmp_fp = tempfile.NamedTemporaryFile('w+')
                json.dump(config_extra, tmp_fp)
                tmp_fp.flush()
                config_files.append(tmp_fp.name)

            conf_object = Configuration({'config': config_files}, RunMode.BACKTEST)
            config = conf_object.get_config()
            validate_config_consistency(config)
            if config.get('strategy'):
                if data_location is None and 'datadir' in config_extra:
                    config['datadir'] = Path(config_extra['datadir'])
                else:
                    config['datadir'] = data_location
                    
                if trading_mode is None and 'trading_mode' not in config_extra:
                    config['trading_mode'] = 'spot'
                    config['candle_type_def'] = CandleType.SPOT
                    del config['margin_mode']
                else:
                    if trading_mode is CandleType.FUTURES:
                        config['trading_mode'] = "futures"
                        config['margin_mode'] = "isolated"
                        config['candle_type_def'] = trading_mode

                config['dataformat_ohlcv'] = data_format
                
                strategy = StrategyResolver.load_strategy(config)
                config['timeframe'] = c.get('timeframe', strategy.timeframe)
                
                if timeframe_detail is not None:
                    config['timeframe_detail'] = timeframe_detail
                
                strategy.dp = DataProvider(config, None)
            configs.append(config)
        finally:
            if tmp_fp:
                tmp_fp.close()

    return configs


def backtest_all(test_config, parallel, cpu_mult=0.66, timeframe_detail=None, data_location=None, data_format="json", trading_mode=CandleType.SPOT):
    # Generate backtest jobs
    configs = prepare_configs(test_config, timeframe_detail=timeframe_detail, data_location=data_location, data_format=data_format, trading_mode=trading_mode)
    backtest_jobs = []

    for i, c in enumerate(test_config):
        for pair_count in c['pair_count']:
            for timerange in c['timeranges']:
                backtest_jobs.append((configs[i], i, c.get('pairlist'), pair_count, timerange))

    if parallel:
        i = 0
        results = []
        while i < len(backtest_jobs):
            job_split = int(multiprocessing.cpu_count() * cpu_mult)
            if job_split > 1:
                tasks = backtest_jobs[i:i+job_split]
            else:
                tasks = backtest_jobs
            with multiprocessing.Pool(max(job_split, 1)) as pool:
                results.extend([job.get() for job in tqdm([pool.apply_async(backtest_one, p) for p in tasks])])
                pool.close()
                pool.join()
            i += len(tasks)
    else:
        results = [backtest_one(*p) for p in tqdm(backtest_jobs)]
    return results


class StrategyTradeInfo:
    def __init__(self):
        self.strategy_name = None
        self.trades = None
        self.qs_trades = None
        self.config = None
        self.config_source = None


class ComparisonInfo:
    def __init__(self):
        self.strategy_infos = []
        self.config = None
        self.config_source = None
        self.pair_count = None


def prepare_results(test_config, results):
    strategy_comparison = {}
    strategy_trades = {}
    strategy_signal_candles = {}

    for i in test_config:
        if "strategy" in i:
            strategy_signal_candles[i['strategy']] = {}

    for r in results:
        pair_count, comparison_stats, raw_trades, config, config_i, processed_dfs = r

        # Per-paircount comparison of different strategies
        try:
            info = strategy_comparison[config_i]
        except KeyError:
            info = strategy_comparison[config_i] = ComparisonInfo()
            info.stats = []
            info.config = config
            info.config_source = test_config[config_i]
            info.pair_count = pair_count
        info.stats.extend(comparison_stats)

        # Per-strategy trades
        for strat, strat_trades in raw_trades.items():
            try:
                info = strategy_trades[config_i]
            except KeyError:
                info = strategy_trades[config_i] = StrategyTradeInfo()
                info.strategy_name = strat
                info.trades = []
                info.config = config
                info.config_source = test_config[config_i]
            info.trades.append(strat_trades)

        # print(processed_dfs[config['strategy']])
        for pair in processed_dfs[config['strategy']].keys():
            if pair not in strategy_signal_candles[config['strategy']]:
                strategy_signal_candles[config['strategy']][pair] = DataFrame()

            if processed_dfs[config['strategy']][pair].shape[0] > 0:
                processed_dfs[config['strategy']][pair].set_index('date', drop=False)
                strategy_signal_candles[config['strategy']][pair] = pd.concat([strategy_signal_candles[config['strategy']][pair], processed_dfs[config['strategy']][pair]], ignore_index=True)
    # Join trade dataframes
    for config_i, info in strategy_trades.items():
        info.trades = pd.concat(info.trades)
        info.qs_trades = trades_freq2quant(info.trades, info.config)

    # TODO: Save merge results of multiple backtests and save them
    # stats = generate_backtest_stats(data, self.all_results, min_date=min_date, max_date=max_date)
    # store_backtest_stats(self.config['exportfilename'], stats)
    return strategy_comparison, strategy_trades, strategy_signal_candles


def extend_metrics(config: dict, metrics: DataFrame, trades: DataFrame, column):
    profit_abs = trades['profit_abs'].sum()
    final_balance = config['dry_run_wallet'] + profit_abs
    gain_on_acc = ((final_balance / config['dry_run_wallet']) - 1) * 100
    metrics.loc[('GOA', column)] = '{:.2f}%'.format(gain_on_acc)
    metrics.loc[('Final Balance', column)] = '{:.2f}'.format(profit_abs)
    return metrics


def print_quant_stats(test_config, strategy_comparison, strategy_trades, table=True, output=None):
    keys = [
        # ("pair_count", "Pairs", "{}"),
        ("date", "Date", "{:s}"),
        ("key", "Strategy", "{:s}"),
        ("profit_mean_pct", "Profit Avg", "{:.2f}"),
        ("profit_sum_pct", "Profit Cum", "{:.1f}"),
        ("profit_total_pct", "Profit %", "{:.1f}"),
        ("profit_total_abs", "Profit Abs", "{:.0f}"),
        ("duration_avg", "Dur Avg", "{:s}"),
        ("wdl", " Win  Draw  Loss  Win%", "{}"),
        ("max_drawdown_account", "DD %", "{:.1f}"),
    ]
    columns = [l for k, l, f in keys]
    figsize = (8, 5)

    if table:
        all_data = []
        if all([test_config[0]['timeranges'] == test_config[i]['timeranges'] for i in range(len(test_config))]):
            # Interleaved
            for i in range(len(strategy_comparison[0].stats)):
                for config_i, info in strategy_comparison.items():
                    row = dict(info.stats[i])
                    hint = info.config_source.get('hint')
                    if not hint:
                        exchange = info.config["exchange"]["name"]
                        hint = f'{info.pair_count}|{exchange}'
                    row['key'] = f"{row['key']}: {hint}"
                    all_data.append([f.format(row[k]) for k, l, f in keys])
            print(tabulate(all_data, columns, tablefmt="pretty", stralign='right'))
        else:
            # Separated
            for config_i, info in strategy_comparison.items():
                all_data = []
                for row in info.stats:
                    row = dict(row)
                    exchange = info.config["exchange"]["name"]
                    row['key'] = f"{row['key']} ({info.pair_count}|{exchange})"
                    all_data.append([f.format(row[k]) for k, l, f in keys])
                print(tabulate(all_data, columns, tablefmt="pretty", stralign='right'))

    bench_info = strategy_trades[0] if len(strategy_trades) > 1 else None
    bench_qs_trades = bench_info.qs_trades if bench_info is not None else None
    for config_i, info in strategy_trades.items():
        compounded = info.config['stake_amount'] == 'unlimited'
        # print(f'### {strategy} ({pair_count} pairs)')
        if len(strategy_trades) > 1 and config_i == 0:
            continue
        
        if info.qs_trades.shape[0] > 0 or bench_qs_trades.shape[0] > 0:
            metrics = qs.reports.metrics(info.qs_trades, bench_qs_trades, trading_year_days=365, compounded=compounded, display=False, internal=True)
            metrics_start = metrics[:metrics.index[3]].copy()
            if len(strategy_trades) > 1:
                metrics_start = extend_metrics(info.config, metrics_start, info.trades, 'Strategy')
                metrics_start = extend_metrics(strategy_trades[0].config, metrics_start, strategy_trades[0].trades, 'Benchmark')
            metrics = pd.concat([metrics_start, metrics[metrics.index[4]:]])

            print(tabulate(metrics, headers="keys", tablefmt='psql'))
            qs.plots.returns(info.qs_trades, bench_qs_trades, figsize=figsize)
            qs.plots.monthly_heatmap(info.qs_trades, figsize=figsize, compounded=compounded, benchmark=bench_info.strategy_name)
            qs.plots.drawdowns_periods(info.qs_trades, figsize=figsize, compounded=compounded)
            if output:
                qs.reports.html(info.qs_trades, bench_qs_trades,
                                title=f'Strategy analysis: {info.strategy_name} vs {bench_info.strategy_name}',
                                output=output.format(strategy=info.strategy_name, benchmark=bench_info.strategy_name))
