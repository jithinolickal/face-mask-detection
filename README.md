### Install necessary packages
```
pip install -r tests/unit_tests/requirements.txt
```

## STEPS

### 1. Generate training data
1. Run `generate_data.py` 2 times (with mask and without mask) - Make sure to rename the file each time
```
pythin generate_data.py
```
2. Provide the number of image data frame that needs to be captured
3. Make sure you have two `.npy` files after running this file (with mask and without mask)


### 2. Testing it live
1. Run
```
python face_mask_detection.py
```