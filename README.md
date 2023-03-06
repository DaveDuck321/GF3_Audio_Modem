# Audio modem for GF3 :)

## Setting up

```sh
python3 -m venv venv && . ./venv/bin/activate # optionally
pip install -r requirements.txt
cd LDPC && make && cd ..
```

## Using the receiver

```sh
python3 receiver.py
```

You can also use: `python3 receiver.py -q` to list audio devices and `python3
receiver.py -d NAME_OR_NUMBER_OF_DEVICE` to pick one
