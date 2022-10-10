#!/bin/bash
for i in {0..50}
do

	python main.py $((2*i)) 0 0&
	python main.py $((2*i)) 0 1&
	python main.py $((2*i)) 0 2&
	python main.py $((2*i)) 0 3&
	python main.py $((2*i)) 1 0&
	python main.py $((2*i)) 1 1&
	python main.py $((2*i)) 1 2&
	python main.py $((2*i)) 1 3&
	python main.py $((2*i+1)) 0 0&
	python main.py $((2*i+1)) 0 1&
	python main.py $((2*i+1)) 0 2&
	python main.py $((2*i+1)) 0 3&
	python main.py $((2*i+1)) 1 0&
	python main.py $((2*i+1)) 1 1&
	python main.py $((2*i+1)) 1 2&
	python main.py $((2*i+1)) 1 3&
	wait
	echo "done"

done

