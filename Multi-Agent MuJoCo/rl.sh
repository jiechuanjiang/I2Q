#!/bin/bash
alpha=(0 1 2 3)
for i in 0 1 2 3
do
	python main.py 0.01 ${alpha[i]}&
	python main.py 0.05 ${alpha[i]}&
	python main.py 0.005 ${alpha[i]}&
	python main.py 0.02 ${alpha[i]}&
	wait
	echo "done"


done


