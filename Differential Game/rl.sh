#!/bin/bash
alpha=(0.0 0.2 2.0 4.0 0.5 1.0 10.0 3.0)
for i in 0 1 2 3 4 5 6 7
do
	python3 main.py ${alpha[i]} 0&
	python3 main.py ${alpha[i]} 1&
	python3 main.py ${alpha[i]} 2&
	python3 main.py ${alpha[i]} 3&
	python3 main.py ${alpha[i]} 4&
	python3 main.py ${alpha[i]} 5&
	python3 main.py ${alpha[i]} 6&
	python3 main.py ${alpha[i]} 7&
	wait
	echo "done"


done


