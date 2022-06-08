# Isolated Testing Environment

The Isolated Testing Environent was developed to quantitativly test combinations of computer vision algorithms through a modular, stripped down (isolated), and automated approach. The results generated are based on sound statistical performance metrics and careful thought went into building a balanced and comprehensive test data set.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries. 

```bash
pip install numpy
pip install opencv
pip install <other dependencies>
```

## Usage

After populating at least one folder with test images, the program can be run in one of two ways. The first is through single batch testing.
In main, specify which batch to run within the run function such as:

```python
def run(batch="Batch2.1"):
    
python ./main.py 
```
The second approach relies on the test.py script which runs through each specified batch, collects results, and outputs the data into results.csv for further analysis.
An accompanying spreadsheet template includes the statistical logic to calculate accuracy, specificity, etc..

```python
batches = ["Batch1.1","Batch2.1",]

python ./test.py
```

## Image processing

Included in this repo is an image processing pipeline that enables modular and exstensable processing on entire datasets.

Currently, the modules that are encluded are as follows:
1. Masking
2. Thresholding
3. Quantization
4. Erosion
5. Dilation

## Further Development

Addtional modules can easily be included to customize this project to your needs.

## License
[MIT](https://choosealicense.com/licenses/mit/)