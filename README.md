# A Comprehensive Benchmark for Clinical Therapeutics Modeling: Targeting Heterogeneity, Irregularity, and Interdependency

Source code for our paper (link forthcoming) defining a benchmark system considering the intrinsic characteristics of EHR data for clinical therapeutics modeling

## Requirements
The code requires Python 3.7 or later. The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.
```bash
pip install -r requirements.txt
```


## Obtaining Data
For the dataset of tasks defined in our benchmark, first you need to have an access to MIMIC-III Dataset which can be requested [here](https://mimic.physionet.org/gettingstarted/access/). 
Once the database has been created, run the data extraction scripts and data pre-processing scripts in order.
```bash
python Dataset/vitals_extraction.py
python Dataset/events_extraction.py
```

## Training and Evaluation

1. In-Hospital Mortality Task (GRU-CTM)
```bash
python Baseline/GRU-CTM.py --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --with-treatment --causal-masking --seed 0
```
2. In-Hospital Mortality Task (GRU-CTM(-Het.))
```bash
python Baseline/GRU-CTM.py --withoutheter --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --with-treatment --seed 0
```

3. In-Hospital Mortality Task (GRU-CTM(-Irr.))
```bash
python Baseline/GRU-CTM.py --withoutirr --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --with-treatment --seed 0
```
4. In-Hospital Mortality Task (GRU-CTM(-Int.))
```bash
python Baseline/GRU-CTM(-Int.).py --withoutint --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --with-treatment --causal-masking --seed 0
```




