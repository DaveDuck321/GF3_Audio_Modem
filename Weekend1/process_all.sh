#!/bin/sh

mkdir -p output

for i in $(find challenge_files -type f ! -name 'channel_impulse.csv'); do
	bin_file="$(python3 demodulate.py "$i")"
	mv "$(python3 process_file.py "$bin_file")" output/
	rm "$bin_file"
done

