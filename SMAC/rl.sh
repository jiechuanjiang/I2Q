#!/bin/bash
alpha=(0.2 0.1 0.3)
for i in 0 1 2
do
    python3 src/main.py --config=iql with lamb=${alpha[i]} --env-config=sc2&
    python3 src/main.py --config=iql with lamb=${alpha[i]} --env-config=sc2&
    python3 src/main.py --config=iql with lamb=${alpha[i]} --env-config=sc2&
    python3 src/main.py --config=iql with lamb=${alpha[i]} --env-config=sc2&
    wait
    echo "done"
done



