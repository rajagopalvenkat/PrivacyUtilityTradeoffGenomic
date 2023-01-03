# GenomicBeacons_SoftPrivacy
#### Enabling Trade-offs in Privacy and Utility in Genomic Data Beacons and Summary Statistics
###### Authors: Rajagopal Venkatesaramani, Zhiyu Wan, Bradley A. Malin, Yevgeniy Vorobeychik

This project was implemented in Python version 3.5.6, and the following packages are dependencies:

	numpy 
	pickle
	pandas

The experiments were run on systems with varying hardware specifications, but on average, code involving the fixed threshold attacker model required about **40-50** GB of RAM, and the adaptive threshold experiments required **90-100 GB** of RAM. Experiment runtime on the complete dataset may be on the order of several hours.

The data files included in the repository are as follows:

* **In_Pop.pkl** - binarized genomic sequences for individuals over whom the beacon is constructed. An entry of 1 represents presence of a minor allele, and 0 denotes absence. 
* **Not_In_Pop.pkl** - binarized genomic sequences for individuals NOT in the beacon. An entry of 1 represents presence of a minor allele, and 0 denotes absence. 
* **AAFs** - Minor allele frequencies for each SNV considered in the study.
* **In_Pop_Beacon.txt** - The first column in this file is the set of beacon responses constructed over the population of interest. The second column shows alternate allele frequencies (which may not be the same as the minor allele frequencies. MAFs are necessarily <= 0.5).
* **LD_keys_0.2** - List of SNVs which show high correlation (minimum Linkage Disequilibrium coefficient of 0.2) with at least one other SNV within a sliding window of 250 SNVs on either side.
* **LD_vals_0.2** - Corresponding highly-correlated SNVs - mapped line-wise to SNVs in **LD_keys_0.2**. This is what we call *N_LD(j)* in the paper.

The code files included in this repository are described below:

* **SPG_B_fix_correl.py** - The SPG-B algorithm for the fixed threshold model, includes an evaluation when the correlations attack is carried out.
* **SPG_B_ada_correl.py** - The SPG-B algorithm for the adaptive threshold model, includes an evaluation when the correlations attack is carried out.
* **mdfc_defense.py** - The SPG-LD algorithm for the fixed threshold model.
* **mdac_defense.py** - The SPG-LD algorithm for the adaptive threshold model.
* **SPG_R_fix.py** - The SPG-R algorithm for the fixed threshold model.
* **SPG_R_ada.py** - The SPG-R algorithm for the adaptive threshold model.

-------------------------------

Rajagopal (Raj) Venkatesaramani

Email: rajagopal@wustl.edu
