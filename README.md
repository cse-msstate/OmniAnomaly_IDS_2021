# OmniAnomaly_IDS_2021
Edited version of OmniAnomaly that allows for epoch-to-epoch testing

# Installation
This code was run using Python 3.6.9 on Ubuntu 18.04 'Bionic Beaver'. If you are having trouble with installation, try using these versions.

1. Download repository:
```
git clone https://github.com/cse-msstate/OmniAnomaly_IDS_2021
```
   - or you can download the zip file from the green "Code" dropdown
 
![image](https://user-images.githubusercontent.com/8016679/130510524-46c99818-8d1a-4176-a931-2d3d22ec3b22.png)

2. Unzip and make a 'results/' and 'data/' directory
3. Install dependencies: 
```
pip3 install -r requirements.txt
```
   - add or remove the '3' as necessary
   - Use sudo if needed
   - Recommended to use a virtual environment


# Running the Script
Verify installation by placing a dataset csv in the 'data/' directory and running: 
```
python3 main.py <dataset-without-'.csv'> [0-3]

# example
python3 main.py nslkdd_100 0
```
- args[1] = name of dataset without '.csv'
- args[2] = fold number to be tested 0,1,2, or 3

Results should be in the 'results/' directory
