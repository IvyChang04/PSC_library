#!/bin/bash
# set -ex

for data in 15000, 30000, 45000, 60000
do
    python JSS_Experiments/Firewall_table3/fixed_n_wo_sampling_ratio.py --methods sc --size ${data}
done

python JSS_Experiments/Firewall_table3/fixed_n_wo_sampling_ratio.py --methods psc --size 60000

for data in 15000, 30000, 45000, 60000
do
    python JSS_Experiments/Firewall_table4/fixed_n_with_sampling_ratio.py --methods psc sc --size ${data}
done