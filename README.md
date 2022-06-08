# ClinTS-HII: A Clinical Time-series Benchmark Targeting Heterogeneity, Irregularity, and Interdependency

Source code for our paper (link forthcoming) defining a benchmark system considering the intrinsic characteristics of EHR data for clinical therapeutics modeling

## Requirements
The code requires Python 3.7 or later. The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.
```bash
pip install -r requirements.txt
```


## Obtaining Data

### 0. Prepare

1. First you need to have an access to MIMIC-III Dataset which can be requested [here](https://mimic.physionet.org/gettingstarted/access/). 
2. Download the MIMIC-III Clinical Database and place the MIMIC-III Clinical Database as either .csv or .csv.gz files somewhere on your local computer.
3. Install Postgres, you need to make sure that Postgres is installed. For installation, please refer to: http://www.postgresql.org/download/

### 1. Create MIMIC-III in a local Postgres database
   Then you can follow the [scripts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres) to create a database to host the MIMIC-III data.  

### 2. Generate datasets
   Once the database has been created, run the data extraction script.
```bash
python Dataset/data_extraction.py
```
After that, read [Data Preprocessing Notebook](Dataset/data-preprocessing.ipynb) for data preprocessing

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




