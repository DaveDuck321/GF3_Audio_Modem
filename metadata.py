import string


def generate_bytes_for_transmission(filename, content: bytes):
    length_of_file = len(content)

    data = bytearray()
    data.extend(length_of_file.to_bytes(4, byteorder="big"))
    data.extend(filename.encode())
    data.extend(b"\0")
    data.extend(content)

    return data


def decode_received_file(data: bytes):
    length = int.from_bytes(data[:4], byteorder="big")
    filename, padded_content = data[4:].split(b"\0", 1)

    sanitized_filename = "".join(
        filter(
            lambda char: char in (string.digits + string.ascii_letters + ".-_"),
            filename.decode("ascii", "ignore"),
        )
    )
    filename = f"output/file_{sanitized_filename}"

    print(f"Read file length {length} bytes")
    print(f"Writing to file: {filename}")

    return filename, padded_content[:length]
