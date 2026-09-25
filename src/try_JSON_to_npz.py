import json
import numpy as np


input_file = "tmp/manual-test/image_00000.network.json"
output_file = "tmp/manual-test/image_00000.npz"


with open(input_file, "r") as f:
    data = json.load(f)


#base_max_bits = np.array(data["base_max_bits"])
base_max_bits = np.array(int(data["base_max_bits"], 2))
base_min_bits = np.array(int(data["base_min_bits"], 2))

print(base_max_bits, base_min_bits)

spatial_delta_log2_normalized = np.array(
    data["spatial_delta_log2_normalized"],
    dtype=np.float32
)

spatial_rank_normalized = np.array(
    data["spatial_rank_normalized"],
    dtype=np.float32
)


np.savez_compressed(
    output_file,
    base_max_bits=base_max_bits,
    base_min_bits=base_min_bits,
    spatial_delta_log2_normalized=spatial_delta_log2_normalized,
    spatial_rank_normalized=spatial_rank_normalized
)