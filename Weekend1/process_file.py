import sys
from pathlib import Path

def process_custom_file_format(input_path: str):
    file_data = open(input_path, 'rb').read()
    path, size, data = file_data.split(b'\0', 2)

    path = Path(path.decode("utf-8"))
    size = int(size.decode("utf-8"))
    data = data[:size]

    output_path = input_path.replace(".bin", "_") + path.name
    open(output_path, 'wb').write(data)
    return output_path

print(process_custom_file_format(sys.argv[1]))
