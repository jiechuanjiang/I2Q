#!/bin/bash
alpha=(0.55 0.6 0.65)
for i in 0 1 2
do
	python3 main.py ${alpha[i]} 0&
	python3 main.py ${alpha[i]} 1&
	python3 main.py ${alpha[i]} 2&
	python3 main.py ${alpha[i]} 3&
	wait
	echo "done"


done


