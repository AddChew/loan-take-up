# Loan Take Up

This repository contains scripts and notebooks to:
- Build machine learning models to predict if a customer would take up a personal loan if the targeted marketing campaign is carried out

## Setup Instructions

The instructions here are required only if you wish to view/run the Jupyter Notebook on your local machine. Otherwise, you can just proceed to ***Viewing Instructions: To view the HTML version of the results***, which does not require any prior setup.

Run setup.sh to setup your conda python environment and install the necessary libraries. This set of instructions assumes that you are using a linux system with conda pre-installed.
```
chmod +x ./setup.sh
./setup.sh
conda activate loan-take-up
```

## Viewing Instructions

### To view the Jupyter Notebook version of the results:

- From the Home Page of the Jupyter Notebook, navigate to and open notebook.ipynb.
- Run all the cells in the notebook from top to bottom (Need to execute this step to view the interactive visualizations).

### To view the HTML version of the results:

- Open notebook.html in your browser.

### To view the helper scripts used in notebook.ipynb:
- Navigate to and open utils/\_\_init\_\_.py

## Files

notebook.ipynb
> - Jupyter Notebook containing the analysis results
> 
notebook.html
> - HTML version of notebook.ipynb
> 
utils/\_\_init\_\_.py
> - Contains helper classes and functions for analysis
> 
requirements.txt
> - Contains the list of dependencies required to run notebook.ipynb and utils/\_\_init\_\_.py
> 
setup.sh
> - Shell script to setup conda python environment and install the necessary libraries in requirements.txt
>
README.md
> - Contains the instructions for viewing the analysis results
>
data/DS_assessment.xlsx
> - Provided dataset for analysis and modelling
>
presentation.pdf
> - Presentation slides
>