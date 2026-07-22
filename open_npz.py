import numpy as np

npz = np.load(
    "outputs/runs/seg_gradcam_outputs_07_12/arrays/0X1A3D565B371DC573_frame0152_8afb4817.npz",
    allow_pickle=True
)

print(npz.files)

print("zero_representation:", npz["zero_representation"])
print("zero_position:", npz["zero_position"])
print("zero_before_normalization:", npz["zero_before_normalization"])
print("zero_after_normalization:", npz["zero_after_normalization"])

for key in [
    "encoder_cams",
    "forward_cams",
    "backward_cams_chronological",
]:
    arr = npz[key]

    print(f"\n{key}")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min:", arr.min())
    print("max:", arr.max())
    print("nonzero:", np.count_nonzero(arr))



