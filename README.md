Card & Krueger Key Replication Results
Author: Gabe Connors
Course: Policy and Program Evaluation
Professor: Tara Sullivan
Date: Fall 2025

Objective:

This project replicates the key findings of Card & Krueger (1994), “Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania.”
The goal is to reproduce the employment effects using the public dataset (public.dat) and compare results to those reported in the original paper.

Folder Structure:

code/ – Python scripts for data prep and analysis
data/ – Raw and cleaned data files
output/ – Regression output in text format
environment.txt – List of installed packages

How to Run:

python code/build_minimal_csv.py
python code/run.py

Outputs:

data/minimal.csv – Clean dataset
output/results.txt – Contains regression results for:
ΔFTE ~ NJ
ΔFTE ~ GAP

Key Results:

Model: ΔFTE ~ NJ Coefficient: +2.75 p-value: 0.04 Interpretation: NJ restaurants increased employment relative to PA

Model: ΔFTE ~ GAP Coefficient: +16.36 p-value: 0.006 Interpretation: Firms most affected by the wage increase had higher employment growth

These results support Card & Krueger’s central conclusion:
“The 1992 New Jersey minimum wage increase did not reduce employment — and may have increased it slightly.”

Replication Notes:

Minor cleaning adjustments were made to handle missing values.
Standard errors are heteroskedasticity-robust (HC1).
Results are consistent in magnitude and direction with the original study.
This project provided practical experience replicating a well-known study in labor economics

Environment:

The file environment.txt includes the Python version and required packages to ensure reproducibility.
