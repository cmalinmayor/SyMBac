import zarr
import os
import numpy as np

filepath = "/nrs/funke/data/darts/220321_ecoli_growth/raw.zarr/fov=2/channel=BF"
zarr_data = zarr.open(filepath, 'r')

root_path = '/nrs/funke/data/darts/synthetic_data'
group_name = 'real_sample_tubes'
target_group_name = '220321_ecoli_growth_tube1.zarr'
target_directory = os.path.join(root_path, group_name, target_group_name)

# Change this depending on each separate crop
corner_1 = (0, 1348, 787)  # (z, x, y) - one corner
corner_2 = (36, 1312, 1187)  # (z, x, y) - opposite corner

# Determine the start and end points
z_min, z_max = sorted([corner_1[0], corner_2[0]])
x_min, x_max = sorted([corner_1[1], corner_2[1]])
y_min, y_max = sorted([corner_1[2], corner_2[2]])

# Ensure coordinates are within the bounds of the image
z_min, z_max = max(0, z_min), min(zarr_data.shape[0], z_max)
x_min, x_max = max(0, x_min), min(zarr_data.shape[2], x_max)
y_min, y_max = max(0, y_min), min(zarr_data.shape[3], y_max)

# Extract the crop using the calculated coordinates
crop = zarr_data[z_min:z_max, 0, y_min:y_max, x_min:x_max]
print(crop)
# Create and save the cropped Zarr dataset
zarr_file = zarr.open(target_directory, mode='w')
zarr_file['phase'] = crop
