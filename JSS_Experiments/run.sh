#!/bin/bash
set -ex
# experiment for table 3
datasets=('Pendigits' 'Letter')
for dataset in "${datasets[@]}"
do
    python table_3/main.py --method sc psc kmeans --dataset ${dataset} --size -1
done

# experiment for table 4
for data in 15000, 30000, 45000, 60000
do
    python Firewall_table4/main.py --methods sc --size ${data}
done

python Firewall_table4/main.py --methods psc --size 60000


# experiment for table 5
for data in 50000, 200000, 1040000
do
    python NIDS_table5/main.py --methods psc --size ${size} --rate 0.9
    python NIDS_table5/main.py --methods sc --size ${size}
done

# experiment for Figure 1

datasets=('noisy_circles' 'noisy_moons' 'varied' 'aniso' 'blobs' 'no_structure')
for dataset in "${datasets[@]}"
do
    python Synthesis_dataset/figure1.py --dataset ${dataset}
done

# experiment for Figure 2
python Synthesis_dataset/figure2.py
