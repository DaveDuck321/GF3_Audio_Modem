def generate_bytes_for_transmission(filename, content: bytes):
    length_of_file = len(content)

    data = bytearray()
    data.extend(length_of_file.to_bytes(4, byteorder='big'))
    data.extend(filename.encode())
    data.extend(b'\0')
    data.extend(content)

    return data

def decode_received_file(data: bytes):
    length = int.from_bytes(data[:4], byteorder='big')
    filename, padded_content = data[4:].split(b'\0', 1)
    print(filename)

    return filename, padded_content[:length]
