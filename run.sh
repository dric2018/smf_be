#!/bin/zsh

#echo "Initiating conda for shell"
#conda init zsh

echo "Activating conda env..."
if source activate smf_be; then
	echo "Conda environment activated successfully."
	echo "Moving into source folder..."
	cd src/

	echo "Starting training job...<train.py>"
	python ./train.py

	echo "Displaying logfile..."
	tail -n 30 ../logs/logs.txt

	echo "Going back to working directory..."
	cd ..
else
	echo "Error: Failed to activate conda environment...Stopping job" >&2
    	exit 1  # Exit the script with an error code
fi
