from sys import argv

def human_readable_compare_files(filename1, filename2):
    bytes1 = open(filename1, 'rb').read()
    bytes2 = open(filename2, 'rb').read()

    return f"Bit  error: {bit_error(bytes1, bytes2)}\n" \
           f"Byte error: {byte_error(bytes1, bytes2)}\n" \
           f"\n" \
           f"Bit  error ignoring length mismatch: {bit_error(bytes1, bytes2, include_length_error=False)}\n" \
           f"Byte error ignoring length mismatch: {byte_error(bytes1, bytes2, include_length_error=False)}\n"

def bit_error(bytes1, bytes2, include_length_error=True):
    length_error = abs(len(bytes1) - len(bytes2))
    errors = 8*length_error if include_length_error else 0
    total = 8*max(len(bytes1), len(bytes2)) # or based on original?
    for b1, b2 in zip(bytes1, bytes2):
        diff = b1 ^ b2
        while diff != 0:
            # branchless programming go brrrr
            errors += diff & 1
            diff >>= 1

    return errors/total

def byte_error(bytes1, bytes2, include_length_error=True):
    length_error = abs(len(bytes1) - len(bytes2))
    errors = length_error if include_length_error else 0
    total = max(len(bytes1), len(bytes2)) # or based on original?
    for b1, b2 in zip(bytes1, bytes2):
        if b1 != b2: errors += 1

    return errors/total

if __name__ == "__main__":
    print(human_readable_compare_files(argv[1], argv[2]))

