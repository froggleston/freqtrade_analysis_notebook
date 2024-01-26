# freqtrade_analysis_notebook

A Jupyter Notebook to make backtesting, analysis and plotting of freqtrade strategies easier

## Issues

The pandas 2.0 release has broken quantstats so pin your pandas install to <2.0 if you want to use the notebook for now.

Awaiting this PR to merge: https://github.com/ranaroussi/quantstats/pull/263

## Prerequisites

Clone this repository onto your machine.

You will need to install jupyter to run this notebook. Please follow any Jupyter installation instructions for your OS or environment.

You will also need extra dependencies for running the notebook. If using an existing freqtrade install (setup.sh script), activate your existing freqtrade venv environment and pip install:

```
cd /path/to/freqtrade
source .venv/bin/activate

pip install -r /path/to/freqtrade_analysis_notebook/requirements.txt
```

### Docker

This has not been tested in any docker environments, so YMMV, but some instructions that might be useful are in [this issue](https://github.com/froggleston/freqtrade_analysis_notebook/issues/1)

## Installation

Follow one of the two methods below:

### Easiest installation

- Copy all `.py` files and the `RollingBacktestNotebook.ipynb` into your base freqtrade folder, **not** the user_data/notebooks folder.
- Set the `freqtrade_dir` variable in the notebook to `"."`

### Easy-ish installation

- Leave the `.py` files and `RollingBacktestNotebook.ipynb` notebook in your cloned directory.
- Set the `freqtrade_dir` to the absolute path of your base freqtrade folder, e.g. `/home/froggleston/freqtrade`
- If you set your `user_data_dir` and `strategy_path` in your config, they need to also be set to the absolute paths to your folders, e.g.
  - `"user_data_dir" : "/home/froggleston/freqtrade/user_data"`
  - `"strategy_path": "/home/froggleston/freqtrade/user_data/strategies"`

## Running

### Via the command line

If you copied the notebook and helper files to your freqtrade folder, navigate there:

```
cd /path/to/freqtrade
```

Or if you left the notebook and helper files in the cloned git folder, navigate there:

```
cd /path/to/freqtrade_analysis_notebook
```

Then run the jupyter server:

```
jupyter notebook --port 8889 --ip 0.0.0.0 --no-browser --NotebookApp.allow_origin='*'
```

The output from the command should contain three links to the Jupyter server.
Pick any of these to open up a new Jupyter file browser tab in your preferred browser.

### Via an IDE

If using an IDE like vscode, install an available jupyter extension and open the freqtrade folder. Then open the ipynb file and run the cells as normal.

## Usage

Use the toolbar at the top of the plot to change behaviour or select/deselect data series.

Pan x only, pan x and y, or drag zoom to move around. The plot will automatically readjust to fit the candles in the current view.

Clicking Reset will zoom back out to the whole configured `plot_tr` timerange.

Click on a data point to view the indicator values in the main plot for that x index. When selected a dashed line will appear through all subplots making it easier to keep track of the selected date index across large numbers of subplots. Double clicking anywhere in the plot removes the highlight.

Mouseover main plot and subplot data series to see individual values.

## Known Issues

### bokeh

- If you scroll wheel in or out too fast it breaks the plot and you have to regenerate it by re-running the cell.
- Click selection requires clicking on a data point and not anywhere on the plot.
- Clicking a data series in the legend will only hide the line plot for that series and not the scatter (which is used for click selection).
- Hover tooltips for the main plot are very difficult to get working and maintain readability, hence the summary table on the right.
- The summary table on the right cannot be auto-sorted, so to get the indicator list for a click selection point, click the value column twice to sort descending. (After you've done this once however, the table maintains that sort order when selecting other data points).
- The signal tooltips appear overlaid on the candle tolltips, making the candle data hard to read.
- No OHLCV data in the summary table.

### plotly

- works but slow
- not as fancy