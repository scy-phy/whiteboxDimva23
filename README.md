# White-box Concealment Attacks Against Anomaly Detectors for Cyber-Physical Systems
## in proceedings at DIMVA 2023

- Each folder contains the whitebox attack code, the attacked model and the resulting adversarial examples
    - all the adversarial examples can be created starting from SWaT dataset available upon request from iTrust Singapore http://itrust.sutd.edu.sg/

- Spoofing Framework folder contains the SWaT data and the black-box adversarial examples generated with the framework by Erba et al. https://github.com/scy-phy/ICS_Generic_Concealment_Attacks

 
- compute_cost_sample.py is the script used to compute the Euclidean and hamming distance

When you use the code from this repository, please cite our work as follows

```
@InProceedings{erba23whitebox,
author="Erba, Alessandro
and Tippenhauer, Nils Ole",
title="White-Box Concealment Attacks Against Anomaly Detectors for Cyber-Physical Systems",
booktitle="Detection of Intrusions and Malware, and Vulnerability Assessment",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="111--131",
isbn="978-3-031-35504-2"
}
```