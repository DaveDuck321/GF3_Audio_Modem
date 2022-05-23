rm -f stats
mkdir -p test-outputs/
for prefix in {10..12}; do
	for body in {13..16}; do
		echo "prefix: 2^$prefix, body: 2^$body" >> stats
		sed -i 's/^OFDM_BODY_LENGTH.*/OFDM_BODY_LENGTH = 1 << '"$body"'/;
		        s/^OFDM_CYCLIC_PREFIX_LENGTH.*/OFDM_CYCLIC_PREFIX_LENGTH = 1 << '"$prefix/" \
			config.py
		# 3 runs
		for i in {1..3}; do
			(sleep 2 && python3 transmitter.py --device pulse ../file-to-send.txt) &
			(sleep 10 && echo) | python3 receiver.py --device pulse
			python3 error_stats.py --mr ../file-to-send.txt output >> stats
			mv output outputs/output_$prefix_$body_$i.txt
		done
		echo >> stats
	done
done

