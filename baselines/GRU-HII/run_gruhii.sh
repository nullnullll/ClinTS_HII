# In the paper, the number of seed is select from [0,1,2]

# Reproduce In-Hospital Mortality Task (GRU-HII)
python baselines/GRU-HII/GRU-HII.py --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --causal-masking --seed 0

# # Reproduce In-Hospital Mortality Task (GRU-HII(-Het.))
# python baselines/GRU-HII/GRU-HII.py --withoutheter --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --seed 0

# # Reproduce In-Hospital Mortality Task (GRU-HII(-Irr.))
# python baselines/GRU-HII/GRU-HII.py --withoutirr --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --least-winsize 0.5 --seed 0

# # Reproduce In-Hospital Mortality Task (GRU-HII(-Int.))
# python baselines/GRU-HII/GRU-HII(-Int.).py --withoutint --task in_hospital_mortality --niters 200 --alpha 5 --lr 0.0001 --batch-size 32 --rec-hidden 128 --num-heads 4 --sample-times 5 --causal-masking --seed 0