from sys import argv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def human_readable_compare_files(filename1, filename2):
    bytes1 = open(filename1, 'rb').read()
    bytes2 = open(filename2, 'rb').read()

    return f"Bit  error: {bit_error(bytes1, bytes2)}\n" \
           f"Byte error: {byte_error(bytes1, bytes2)}\n" \
           f"\n" \
           f"Bit  error ignoring length mismatch: {bit_error(bytes1, bytes2, include_length_error=False)}\n" \
           f"Byte error ignoring length mismatch: {byte_error(bytes1, bytes2, include_length_error=False)}\n"

def machine_readable_compare_files(filename1, filename2):
    bytes1 = open(filename1, 'rb').read()
    bytes2 = open(filename2, 'rb').read()
    return f"{bit_error(bytes1, bytes2)}\t{byte_error(bytes1, bytes2)}\t" \
           f"{bit_error(bytes1, bytes2, include_length_error=False)}\t" \
           f"{byte_error(bytes1, bytes2, include_length_error=False)}"

def get_cumulative_bit_errors(bytes1: bytes, bytes2: bytes):
    return np.cumsum(np.unpackbits(np.array(list(bytes1), dtype=np.uint8)) != np.unpackbits(np.array(list(bytes2), dtype=np.uint8)))

def plot_cumulative_error(bytes1: bytes, bytes2: bytes):
    assert len(bytes1) == len(bytes2)
    cumulative_errors = get_cumulative_bit_errors(bytes1, bytes2)

    # Fit the first 20% to see linear trends
    final_index = len(cumulative_errors) // 5
    best_fit = np.polyfit(np.linspace(0, final_index, final_index), cumulative_errors[:final_index], 1)
    t = np.linspace(0, len(cumulative_errors), 2)



    plt.title("Cumulative bit errors")
    plt.xlabel("Total bits processed")
    plt.ylabel("Total bit errors")
    plt.plot(t, np.poly1d(best_fit)(t), label="Linear 10% trend")
    plt.plot(cumulative_errors, label="Demodulation errors")
    # plt.axvline(x = 0, ls='--')
    # plt.axvline(x = final_index, ls='--')
    plt.vlines(range(0, 8*len(bytes1), 1534*128//2), 0, cumulative_errors[-1], color='C2')
    plt.legend()
    plt.savefig('cumulative_bit_error.pgf')
    plt.savefig('cumulative_bit_error.pdf')

def bit_error(bytes1, bytes2, include_length_error=True):
    if include_length_error:
        length_error = abs(len(bytes1) - len(bytes2))
        errors = 8*length_error
        total = 8*max(len(bytes1), len(bytes2))
    else:
        errors = 0
        total = 8*min(len(bytes1), len(bytes2))

    for b1, b2 in zip(bytes1, bytes2):
        diff = b1 ^ b2
        while diff != 0:
            # branchless programming go brrrr
            errors += diff & 1
            diff >>= 1

    return errors/total

def byte_error(bytes1, bytes2, include_length_error=True):
    if include_length_error:
        length_error = abs(len(bytes1) - len(bytes2))
        errors = length_error
        total = max(len(bytes1), len(bytes2)) # or based on original?
    else:
        errors = 0
        total = min(len(bytes1), len(bytes2))

    for b1, b2 in zip(bytes1, bytes2):
        if b1 != b2: errors += 1

    return errors/total

if __name__ == "__main__":
    if argv[1] == "--mr":
        print(machine_readable_compare_files(argv[2], argv[3]))
    else:
        print(human_readable_compare_files(argv[1], argv[2]))
