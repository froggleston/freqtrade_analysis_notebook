class PlotHelper():
    def __init__(self, plotlib="bokeh", timeframe="5m") -> None:
        if "plotly" == plotlib:
            try:
                import plotly.graph_objects as go
                from plotly_plotter import PlotlyPlotter
                self.plotter = PlotlyPlotter()            
            except ImportError:
                logger.exception("Module plotly not found \n Please install using `pip install plotly`")
                exit(1)
        else:
            try:
                from bokeh.plotting import figure
                from bokeh_plotter import BokehPlotter
                self.plotter = BokehPlotter(timeframe=timeframe)
            except ImportError:
                logger.exception("Module bokeh not found \n Please install using `pip install bokeh`")
                exit(1)
