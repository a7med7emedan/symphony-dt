# SYMPHONY results explorer

## How to run

1. Open `symphony_explorer.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run all cells from top to bottom (Kernel > Restart and Run All, or Runtime > Run all in Colab).
   The hidden setup cells execute silently and take about 30 seconds depending on download speed.
3. When execution reaches section E, click the **Run SYMPHONY explorer** button.
4. The clinical dashboard will appear with patient selection controls,
   a tabbed explorer (profile, trajectory, risk, reliability), and a clinical
   reasoner console with insight cards and free-text query support.

## What the user sees

After clicking the workflow button, the visible output consists of:

1. A one-line bundle summary (run ID, data source, fold count).
2. A convergence status indicator.
3. The SYMPHONY clinical explorer: dropdown selectors for cohort, patient, event,
   and endpoint, with four tabbed views (patient profile, disease trajectory,
   risk estimates, model reliability).
4. The clinical reasoner console: insight cards for common clinical questions
   and a free-text input for arbitrary queries. All answers are deterministic
   and trace to specific source files in the bundle.

Optionally, section F provides buttons for four diagnostic deep dives:
cross-fold stability, inference health, translational indices audit, and
worked examples. These produce detailed tables and plots when clicked.

## Where outputs are saved

This notebook renders all results inline. No external files are written.
Figures can be saved manually via right-click in the browser.

## How to enable diagnostic details

1. Scroll to section F (below the main dashboard).
2. Click the expandable "Diagnostics and reproducibility details" summary.
3. Click any of the four diagnostic buttons to generate the corresponding analysis.
4. To inspect hidden infrastructure cells (data loading, validation, manifests),
   expand the collapsed cells in sections B through D by clicking the cell's
   disclosure triangle in the notebook sidebar.

## Dependencies

The notebook requires Python 3.8 or later with the following packages
(all standard in Colab and most scientific Python environments):

    numpy, pandas, matplotlib, ipywidgets, pyyaml

pyyaml is installed automatically if not already present.
The results bundle is not shipped. Point `ZIP_PATH` at your own `SYMPHONY_results.zip`,
upload it in Colab, or set `SYMPHONY_BUNDLE_GDRIVE_ID` to a Drive file you are allowed to share.

## File structure

    notebooks/explorer/
        symphony_explorer.ipynb   Main notebook
        README.md                 This file
