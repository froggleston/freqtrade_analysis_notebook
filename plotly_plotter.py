import sys,traceback

import matplotlib.pyplot as plt
from freqtrade.plot.plotting import generate_candlestick_graph
import plotly.offline as pyo

import pandas as pd  # noqa
from pandas import DataFrame, Series
from datetime import datetime, timedelta

from math import pi

class PlotlyPlotter():
    def __init__(self, backend="svg") -> None:
        self.backend = backend
        pyo.init_notebook_mode(connected=True)

    def do_plot(self, pair, data, trades, d_start, d_end, plot_config=None, buy_tags=None, sell_tags=None, width=1400, height=1200):
        try:
            trades_red = pd.DataFrame()
            
            if trades.shape[0] > 0:
                # Filter trades to one pair
                trades_red = trades.loc[trades['pair'] == pair].copy()
            
            buyf = data[data.filter(regex=r'^enter', axis=1).values==1]

            if buyf.shape[0] > 0 and trades_red.shape[0] > 0:
                for t, v in trades_red.open_date.items():
                    tc = buyf.loc[(buyf['date'] < v)]
                    if tc is not None and tc.shape[0] > 0:
                        bt = tc.iloc[-1].filter(regex=r'^enter', axis=0)
                        bt.dropna(inplace=True)
                        tbt = trades_red.loc[t, 'enter_tag']
                        tst = trades_red.loc[t, 'exit_reason']
                        
                        if isinstance(tbt, Series):
                            tbt = tbt.iloc[0]
                        if isinstance(tst, Series):
                            tst = tst.iloc[0]
                            
                        if buy_tags is not None and tbt not in buy_tags and t in trades_red:
                            trades_red.drop(t, inplace=True)
                        else:
                            trades_red.loc[t, 'exit_reason'] = f"{tbt} / {trades_red.loc[t, 'exit_reason']}"
                            
                        if sell_tags is not None and tst not in sell_tags and t in trades_red:
                            trades_red.drop(t, inplace=True)
                        else:
                            trades_red.loc[t, 'exit_reason'] = f"{tst} / {trades_red.loc[t, 'exit_reason']}"

            # Limit graph period to your BT timerange
            data_red = data[d_start:d_end]
            graph = generate_candlestick_graph(pair=pair,
                                                data=data_red,
                                                trades=trades_red,
                                                plot_config=plot_config
                                                )

            graph.update_layout(autosize=False,width=width,height=height)
            pyo.iplot(graph, show_link = False)
            
        except Exception as e:
            traceback.print_exc(*sys.exc_info())
            print("You got frogged: ", e)