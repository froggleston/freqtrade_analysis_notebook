import sys, traceback

from bokeh import events
from bokeh.core.enums import MarkerType
from bokeh.events import DoubleTap, Tap
from bokeh.layouts import column, row
from bokeh.models import (Band, BoxAnnotation,
                          CrosshairTool, ColumnDataSource, CustomJS,
                          DataTable, DateFormatter, Div,
                          HoverTool, HStrip,
                          NumeralTickFormatter, Range1d,
                          Scatter, Span,
                          TableColumn, TapTool, VStrip)
from bokeh.plotting import figure as _figure
from bokeh.plotting import show
from bokeh.io import output_notebook, curdoc
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap

import numpy as np
import pandas as pd  # noqa
from pandas import DataFrame, Series
from datetime import datetime, timedelta
from functools import partial
from collections import OrderedDict

from math import pi

class BokehPlotter():
    def __init__(self, backend="webgl", timeframe="5m") -> None:
        self.backend = backend
        self.timeframe = timeframe
        self.glyphmap = {
            "enter_long":{"marker":"triangle","fill_color":"green"},
            "exit_long":{"marker":"inverted_triangle","fill_color":"red"},
            "enter_short":{"marker":"inverted_triangle","fill_color":"blue"},
            "exit_short":{"marker":"triangle","fill_color":"fuchsia"}
        }

        output_notebook()

    def do_plot(self, pair: str, data: pd.DataFrame, trades: pd.DataFrame,
                d_start: datetime, d_end: datetime,
                plot_config: dict = None,
                buy_tags: list = None,
                sell_tags: list = None,
                width: int = 1400, height: int = 1200):
        try:
            trades_red = pd.DataFrame()

            if trades.shape[0] > 0:
                # Filter trades to one pair
                trades_red = trades.loc[trades['pair'] == pair].copy()

            buyf = data[data.filter(regex=r'^enter', axis=1).values==1].copy()

            data["plot_cumprof"] = Series(np.nan).copy()
            data["plot_cumprof"].iloc[0] = 0

            if buyf.shape[0] > 0 and trades_red.shape[0] > 0:
                curr_profit = 0
                for t, v in trades_red.open_date.items():
                    curr_profit = curr_profit + trades_red.loc[t, 'profit_abs']

                    if v in data:
                        data.at[v, 'plot_cumprof'] = curr_profit

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
                                trades_red.loc[t, 'exit_reason'] = \
                                    f"{tbt} / {trades_red.loc[t, 'exit_reason']}"

                            if sell_tags is not None and tst not in sell_tags and t in trades_red:
                                trades_red.drop(t, inplace=True)
                            else:
                                trades_red.loc[t, 'exit_reason'] = \
                                    f"{tst} / {trades_red.loc[t, 'exit_reason']}"

            data['plot_cumprof'].ffill(inplace=True)

            # Limit graph period to your BT timerange
            data_red = data[d_start:d_end]

            figs, dt = self._generate_bokeh_candlestick_graph(pair=pair,
                                                              data=data_red,
                                                              trades=trades_red,
                                                              plot_config=plot_config
                                                              )

            show(row(gridplot(figs), dt))

        except Exception as e:
            traceback.print_exc(*sys.exc_info())
            print("You got frogged: ", e)

    def _get_custom_glyphs(self, df: pd.DataFrame, column_name, marker, size=10, fill_color=None):
        if column_name in df.columns:
            glyph_df = df[df[column_name] == 1]
            if len(glyph_df) > 0:
                data = dict(x=glyph_df.date, y=glyph_df.close)

                marker_source = ColumnDataSource(data=data)
                glyphs = Scatter(
                    x='x',
                    y='y',
                    marker=marker,
                    name=column_name,
                    size=size,
                    fill_color=fill_color,
                )

                return marker_source, glyphs
        return None, None

    def _get_signal_glyphs(self, df: pd.DataFrame, column_name, fill_color=None):
        if fill_color is not None:
            self.glyphmap[column_name]['fill_color'] = fill_color

        direction = column_name.split("_")[0]

        if column_name in df.columns:
            glyph_df = df[df[column_name] == 1]
            if len(glyph_df) > 0:
                data = dict(x=glyph_df.date, y=glyph_df.close, signal_type=glyph_df[column_name])
                if f"{direction}_tag" in glyph_df:
                    data[f"{direction}_tag"] = glyph_df[f"{direction}_tag"]

                signal_source = ColumnDataSource(data=data)
                glyphs = Scatter(
                    x='x',
                    y='y',
                    marker=self.glyphmap[column_name]['marker'],
                    name=column_name,
                    size=12,
                    fill_color=self.glyphmap[column_name]['fill_color'],
                )

                return signal_source, glyphs
        return None, None

    def _get_trade_entry_glyphs(self, trades: pd.DataFrame):
        if trades is not None and len(trades) > 0:
            trades['desc'] = trades.apply(
                lambda row: f"{row['profit_ratio']:.2%}, " +
                (f"{row['enter_tag']}, " if row['enter_tag'] is not None else "") +
                f"{row['exit_reason']}, " +
                f"{row['trade_duration']} min",
                axis=1)

            data = dict(x=trades["open_date"], y=trades["open_rate"], desc=trades['desc'])
            trades_source = ColumnDataSource(data=data)
            glyphs = Scatter(
                x='x',
                y='y',
                marker="circle",
                name="Trade entry",
                size=12,
                fill_color="cyan",
            )
            return trades_source, glyphs
        return None, None

    def _get_trade_exit_glyphs(self, trades: pd.DataFrame):
        if trades is not None and len(trades) > 0:
            trades = trades.loc[trades['profit_ratio'] > 0].copy()
            trades['desc'] = trades.apply(
                lambda row: f"{row['profit_ratio']:.2%}, " +
                (f"{row['enter_tag']}, " if row['enter_tag'] is not None else "") +
                f"{row['exit_reason']}, " +
                f"{row['trade_duration']} min",
                axis=1)

            data = dict(x=trades.loc[trades['profit_ratio'] > 0, "close_date"],
                        y=trades.loc[trades['profit_ratio'] > 0, "close_rate"],
                        desc=trades['desc'])
            trades_source = ColumnDataSource(data=data)
            glyphs = Scatter(
                x='x',
                y='y',
                marker="square",
                name="Trade exit",
                size=12,
                fill_color="green",
            )
            return trades_source, glyphs
        return None, None

    def _get_trade_loss_glyphs(self, trades: pd.DataFrame):
        trades = trades.loc[trades['profit_ratio'] <= 0].copy()
        if trades is not None and len(trades) > 0:
            trades['desc'] = trades.apply(
                lambda row: f"{row['profit_ratio']:.2%}, " +
                (f"{row['enter_tag']}, " if row['enter_tag'] is not None else "") +
                f"{row['exit_reason']}, " +
                f"{row['trade_duration']} min",
                axis=1)

            data = dict(x=trades.loc[trades['profit_ratio'] <= 0, "close_date"],
                        y=trades.loc[trades['profit_ratio'] <= 0, "close_rate"],
                        desc=trades['desc'])
            trades_source = ColumnDataSource(data=data)
            glyphs = Scatter(
                x='x',
                y='y',
                marker="square",
                name="Trade exit",
                size=12,
                fill_color="red",
            )
            return trades_source, glyphs
        return None, None

    def _get_box_spans(self, fig, df: pd.DataFrame, column_name, fill_color='#50C878'):
        if column_name in df.columns:
            glyph_df = df.loc[df[column_name] == 1]
            if len(glyph_df) > 0:
                period = pd.Timedelta(self.timeframe)
                dt = glyph_df['date']
                in_block = ((dt - dt.shift(-1)).abs() == period) | (dt.diff() == period)
                filt = glyph_df.loc[in_block]
                breaks = filt['date'].diff() != period
                groups = breaks.cumsum()

                x0s = []
                x1s = []
                for _, frame in filt.groupby(groups):
                    x0s.append(frame.index[0])
                    x1s.append(frame.index[1])

                fig.vstrip(fill_color=fill_color, fill_alpha=0.2, line_color=fill_color, line_alpha=0.4,
                           x0=x0s, x1=x1s)

    ## modified from https://github.com/ndepaola/bokeh-candlestick
    def _generate_bokeh_candlestick_graph(self,
                                          pair: str,
                                          data: pd.DataFrame = DataFrame(),
                                          trades: pd.DataFrame = None,
                                          plot_config: dict = None,
                                          width: int = 1280, height: int = 720,
                                          tools="xpan,pan,xwheel_zoom,box_zoom,reset,save",
                                          backend="webgl"):
        ### SETUP
        xaxis_dt_format = '%d %b %Y'
        if data['date'][0].hour > 0:
            xaxis_dt_format = '%d %b %Y, %H:%M:%S'

        date_formatter = '%d %b %Y, %H:%M:%S'

        # Colour scheme for increasing and descending candles
        GREEN = '#50C878'
        RED = '#FF2400'
        CANDLE_COLOURS = [GREEN, RED]

        bar_width = pd.Timedelta(self.timeframe)

        OHLCV_FILTER = OrderedDict((
            ('open', 'first'),
            ('high', 'max'),
            ('low', 'min'),
            ('close', 'last'),
            ('volume', 'sum'),
        ))

        df = data[list(OHLCV_FILTER.keys())].copy(deep=False)
        ohlc_minmax_values = df[['high', 'low']].copy(deep=False)
        index = df.index

        ### MAIN PLOT
        bokeh_fig = partial(
            _figure,
            tools=tools,
            active_drag='xpan',
            active_scroll='xwheel_zoom',
            x_axis_type='datetime',
            title=pair,
            width=width,
            height=height,
            output_backend=backend
        )
        pad = (index[-1] - index[0]) / 20

        _kwargs = dict(x_range=Range1d(
            index[0], index[-1],
            min_interval=10,
            bounds=(index[0] - pad, index[-1] + pad))) if index.size > 1 else {}

        fig = bokeh_fig(**_kwargs)

        source = ColumnDataSource(df)
        source.add((df['open'] >= df['close']).values.astype(np.uint8).astype(str), 'green')

        price_formatter = "0[.]00[000]f"
        fig.yaxis[0].formatter = NumeralTickFormatter(format=price_formatter)

        # ## setup tooltip formatting
        h1_tooltips = [("Open", "@open{" + price_formatter + "}"),
                       ("High", "@high{" + price_formatter + "}"),
                       ("Low", "@low{" + price_formatter + "}"),
                       ("Close", "@close{" + price_formatter + "}"),
                       ("Date", "@date{" + date_formatter + "}")]

        H3_TOOLTIPS = """
            <div>
                <div>
                    <span style="font-size: 10px; font-weight: bold;">$name</span>:
                    <span style="font-size: 10px; color: #696;">$snap_y{0[.]00[000]f}</span>
                </div>
            </div>
        """

        ### DO PLOTTING
        candle_colours = factor_cmap('green', CANDLE_COLOURS, ['0', '1'])

        fig.segment(x0='date', y0='high', x1='date', y1='low', source=source, color=candle_colours)
        r = fig.vbar(x='date', width=bar_width, top='open', bottom='close', source=source,
                     line_color="lightgrey", fill_color=candle_colours)

        h1 = HoverTool(
            description="Toggle Candle Tooltips",
            renderers=[r],
            tooltips=h1_tooltips,
            formatters={
                '@date': 'datetime'
            },
            mode="mouse")

        # Set up the hover tooltip to display some useful data
        fig.add_tools(h1)

        source.add(ohlc_minmax_values.min(1), 'ohlc_low')
        source.add(ohlc_minmax_values.max(1), 'ohlc_high')
        source.add(df.index, 'index')

        # add clickable points and vertical span line
        span_select_src = ColumnDataSource(data={
            'x': [],
            'y': []
        })
        span_select_r = fig.scatter(x="x",
                                    y="y",
                                    size=10,
                                    fill_color = "blue",
                                    source=span_select_src)
        span = Span(dimension='height', line_dash="dashed", line_width=2)
        fig.add_layout(span)

        select_js_cb_code = '''
            var idx = 0;
            var name = "NONE";

            var span_x = null;

            const x = [];
            const y = [];
            const ind = [];

            for (var ls in dotsrcs) {
                if (dotsrcs[ls].selected.indices.length != 0) {
                    idx = dotsrcs[ls].selected.indices;
                    break;
                }
            }

            if (idx.length != 0 && idx != 0) {
                var d = mainsrc.data;
                for (var i in d) {
                    var didx = mainsrc.data['x'][idx[0]];
                    span_x = didx;

                    x.push(didx);

                    if (i == "x") {
                        y.push(new Date(mainsrc.data[i][idx[0]]));
                        ind.push("date");
                    }
                    else {
                        y.push(mainsrc.data[i][idx[0]]);
                        ind.push(i);
                    }
                }

                span.location = span_x;
                span.visible = true;

                span_select_src.data['x'] = x;
                span_select_src.data['y'] = y;
                span_select_src.data['ind'] = ind;
            }
            else {
                span.visible = false;
            }

            span_select_src.change.emit();
        '''

        unselect_js_cb_code = '''
            span.visible = false;
            span_select_src.data['x'] = [];
            span_select_src.data['y'] = [];
            span_select_src.data['ind'] = [];

            span_select_src.change.emit();
        '''

        autoscale_js_cb_code = '''
            if (!window._bt_scale_range) {
                window._bt_scale_range = function (range, min, max, pad) {
                    "use strict";
                    if (min !== Infinity && max !== -Infinity) {
                        pad = pad ? (max - min) * .03 : 0;
                        range.start = min - pad;
                        range.end = max + pad;
                    } else console.error('scale range error:', min, max, range);
                };
            }

            clearTimeout(window._bt_autoscale_timeout);

            window._bt_autoscale_timeout = setTimeout(function () {
                /**
                * @variable cb_obj `fig_ohlc.x_range`.
                * @variable source `ColumnDataSource`
                * @variable ohlc_range `fig_ohlc.y_range`.
                * @variable volume_range `fig_volume.y_range`.
                */
                "use strict";

                /* bar width timeframe millis to minutes */
                var coeff = 1000 * 60 * Math.floor(bar_width / 60000)

                var startdate = new Date(Math.round(Math.floor(cb_obj.start) / coeff) * coeff).getTime()
                var enddate = new Date(Math.round(Math.ceil(cb_obj.end) / coeff) * coeff).getTime()

                let i = source.data.index.indexOf(startdate),
                    j = source.data.index.indexOf(enddate)

                let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j)),
                    min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
                _bt_scale_range(ohlc_range, min, max, true);

            }, 50);

        '''

        mainplot_renderers = []
        dot_renderers = []
        linesrcs = {}
        dotsrcs = {}

        plot_cols = ['date','open','close','high','low','volume']

        main_source = ColumnDataSource(data={
            'x': data.index,
        })

        # select only columns from the main dataframe for plotting
        if plot_config is not None:
            if 'main_plot' in plot_config:
                for k, v in plot_config['main_plot'].items():
                    if k in data:
                        plot_cols.append(k)

        main_plot_data = data[plot_cols]
        main_plot_data.set_index('date', inplace=True, drop=False)

        # iterate over main_plot plot_config
        if plot_config is not None:
            if 'main_plot' in plot_config:
                for k, v in plot_config['main_plot'].items():
                    if "marker" in v:
                        custom_marker_source, custom_marker_glyphs = self._get_custom_glyphs(
                            main_plot_data,
                            k,
                            v['marker'],
                            fill_color=v['color']
                        )

                        if custom_marker_source is not None and custom_marker_glyphs is not None:
                            custom_r = fig.add_glyph(custom_marker_source, custom_marker_glyphs)
                            custom_r_hover = HoverTool(
                                description=f"Toggle {k} Tooltips",
                                renderers=[custom_r],
                                tooltips=[(k, f"@y"+"{"+price_formatter+"}")]
                            )
                            fig.add_tools(custom_r_hover)
                    elif "box" in v:
                        self._get_box_spans(
                            fig,
                            main_plot_data,
                            k,
                            fill_color=v['box']
                        )
                    else:
                        line_source = ColumnDataSource(data={
                            'x1': data.index,
                            'y1': main_plot_data[k]
                        })

                        dot_source = ColumnDataSource(data={
                            'x1': data.index,
                            'y1': main_plot_data[k],
                        })

                        linesrcs[k] = line_source
                        dotsrcs[k] = dot_source
                        main_source.data[k] = main_plot_data[k]

                        # add plot line
                        mainline = fig.line(x='x1',
                                            y='y1',
                                            source=line_source,
                                            name=k,
                                            line_color=v['color'],
                                            line_width=1,
                                            legend_label=k)
                        mainplot_renderers.append(mainline)

                        dot = fig.scatter(x='x1',
                                            y='y1',
                                            source=dot_source,
                                            marker='dot',
                                            color=v['color'],
                                            size=10,
                                            name=f"{k}_dots",
                                            hit_dilation=1.5)

                        dot_renderers.append(dot)

        dot_callback = CustomJS(
            args = {
                'span_select_src': span_select_src,
                'mainsrc': main_source,
                'dotsrcs': dotsrcs,
                'span':span
            },
            code = select_js_cb_code
        )
        hover_dot = TapTool(
            description = f"Tap Select",
            renderers = dot_renderers,
            callback=dot_callback
        )
        fig.add_tools(hover_dot)

        fig.js_on_event(DoubleTap, CustomJS(
            args = {
                'span_select_src' : span_select_src,
                'span':span
            },
            code = unselect_js_cb_code
        ))

        fig.x_range.js_on_change('end', CustomJS(
            args = {
                'ohlc_range': fig.y_range,
                'source': source,
                'bar_width': bar_width,
            },
            code = autoscale_js_cb_code
        ))

        ## add trades
        trade_entry_source, trade_entry_glyphs = self._get_trade_entry_glyphs(trades)
        if trade_entry_source is not None and trade_entry_glyphs is not None:
            tg_r = fig.add_glyph(trade_entry_source, trade_entry_glyphs)
            tg_r_hover = HoverTool(description="Toggle Trade Entry Tooltips",
                                   renderers=[tg_r],
                                   tooltips=[('Info', '@desc')])
            fig.add_tools(tg_r_hover)

        trade_exit_source, trade_exit_glyphs = self._get_trade_exit_glyphs(trades)
        if trade_exit_source is not None and trade_exit_glyphs is not None:
            tg_r = fig.add_glyph(trade_exit_source, trade_exit_glyphs)
            tg_r_hover = HoverTool(description="Toggle Trade Profit Tooltips",
                                   renderers=[tg_r],
                                   tooltips=[('Info', '@desc')])
            fig.add_tools(tg_r_hover)

        trade_loss_source, trade_loss_glyphs = self._get_trade_loss_glyphs(trades)
        if trade_loss_source is not None and trade_loss_glyphs is not None:
            tg_r = fig.add_glyph(trade_loss_source, trade_loss_glyphs)
            tg_r_hover = HoverTool(description="Toggle Trade Loss Tooltips",
                                   renderers=[tg_r],
                                   tooltips=[('Info', '@desc')])
            fig.add_tools(tg_r_hover)

        ## add signals
        for signal in self.glyphmap.keys():
            direction = signal.split("_")[0]
            signal_source, signal_glyphs = self._get_signal_glyphs(data, signal)
            if signal_source is not None and signal_glyphs is not None:
                sg_r = fig.add_glyph(signal_source, signal_glyphs)
                sg_r_hover = HoverTool(
                    description=f"Toggle {direction} Signal Tooltips",
                    renderers=[sg_r],
                    tooltips=[('Type', signal), (f"{direction} Tag", f"@{direction}_tag")])
                fig.add_tools(sg_r_hover)

        ## auto add bollingers if they exist
        if "bb_upperband" in data and "bb_lowerband" in data:
            bb_source = ColumnDataSource(data={
                'x': data.index,
                'bbl': data.bb_lowerband,
                'bbu': data.bb_upperband,
            })
            band = Band(base="x", lower="bbl", upper="bbu", source=bb_source,
                        fill_alpha=0.3, fill_color="powderblue", line_color="powderblue")
            fig.add_layout(band)

        ### SUBPLOTS
        vol_fig = _figure(
            title="Volume",
            x_axis_type="datetime",
            tools="xpan,pan,xwheel_zoom",
            toolbar_location=None,
            width=width,
            height=200,
            x_range=fig.x_range)
        vol_fig.grid.grid_line_alpha=0.3
        vol_fig.vbar(data.date, bar_width, data.volume, [0]*data.shape[0])
        vol_fig.add_layout(span)

        allfigs = [[fig],[vol_fig]]

        if plot_config is not None:
            if 'subplots' in plot_config:
                for name, subplot in plot_config['subplots'].items():
                    plot_ok = False
                    sub_fig = _figure(
                        title=name,
                        x_axis_type="datetime",
                        tools="xpan,pan,xwheel_zoom",
                        toolbar_location=None,
                        width=width,
                        height=200,
                        x_range=fig.x_range)
                    sub_tooltips = [("Date", "@x{" + date_formatter + "}")]
                    line_source = ColumnDataSource(data={
                        'x': data.index,
                    })
                    for k, v in subplot.items():
                        if k in data:
                            line_source.data[k] = data[k]
                            sub_tooltips.append(
                                (k, f"@{k}"+"{"+price_formatter+"}")
                            )
                            plot_ok = True
                        else:
                            print(f"Warning: Cannot plot {k} - not in main dataframe.")

                    # only add subplot if there is at least one indicator
                    # in the subplot that exists in the dataframe
                    if plot_ok:
                        for k, v in line_source.data.items():
                            if "x" != k:
                                sbl = sub_fig.line(x='x',
                                                   y=k,
                                                   source=line_source,
                                                   line_color=subplot[k]['color'],
                                                   legend_label=k)

                                sub_hover = HoverTool(
                                    description=f"Toggle {name} Tooltips",
                                    renderers=[sbl],
                                    tooltips=sub_tooltips,
                                    formatters={
                                        '@x': 'datetime',
                                    },
                                    mode="mouse")

                                sub_fig.add_tools(sub_hover)

                        sub_fig.add_layout(span)
                        allfigs.append([sub_fig])

        # Add date labels to x axis
        fig.xaxis.major_label_overrides = {
            i: date.strftime(xaxis_dt_format) for i, date in enumerate(
                pd.to_datetime(data["date"])
            )
        }

        # add legend
        fig.legend.location = "top_left"
        fig.legend.click_policy="hide"

        # set theme
        curdoc().theme = "caliber"

        ### Indicator value datatable
        div_columns = [
                TableColumn(field="ind", title="Indicator"),
                TableColumn(field="y", title="Value"),
            ]
        data_table = DataTable(source=span_select_src,
                               columns=div_columns,
                               width=500,
                               # autosize_mode="fit_columns",
                               selectable=False,
                               reorderable=False,
                               height_policy="max")

        return allfigs, data_table
