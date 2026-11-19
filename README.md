# nov26_benchmark_results

This directory contains the code necessary for running and analyzing the performance 
testing benchmarks.

## Contents

The *-runner.ipynb files are Jupyter notebooks used for collecting data for each 
respective test

'color_code' folder contains code for color code memory experiments

The 'clops_v' directory contains the files needed for running the CLOPSv experiment
on IBM Quantum systems

The 'mgym' directory contains a collection of metriq-gym "suites" used for collecting 
LR-QAOA data via the metriq-gym framework

'qv' folder contains circuits and test runner needed for reproducing the Quantum
Volume results

The 'results' directory contains all the result files and additonal scripts and Jupyter 
notebooks needed to analyze and process the results  

## Notes

The Randomized Benchmarking (RB) data is gathered from automated calibration scripts run 
by the devices.  As such, there are no corresponding Jupyter notebooks or test scripts 
for this data.

## License

(C) Copyright IBM 2026.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
