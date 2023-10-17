# LLIE

Implementation of 2 different algorithms:
- Zero-DCE for low-light image enhancement [[link](https://arxiv.org/pdf/2001.06826.pdf)]
- Shallow camera pipeline for night photography rendering (IVL) [[link](https://arxiv.org/pdf/2204.08972.pdf)]

The first one is an efficient unsupervised deep learning method, while the last one is an algorithm based on traditional image processing techniques.

These algorithm were tested on the [LOL](https://daooshee.github.io/BMVC2018website/) dataset, which you need to download (under the Desktop dir) in order to train Zero-DCE and test them.

### Requirements
The code was tested with Python 3.10.12.

Install requirements:
```bash
pip install -r requirements.txt
```

### Test IVL
In order to test this algorithm:

**N.B.** Change the path of the image to be tested in `methods/ivl.py` file.
``` bash
cd methods/
python ivl.py
```

### Train Zero-DCE
The training will save checkpoints under a *ckpt/* folder under DCE directory.

**N.B.** Change the path to the dataset in `methods/DCE/main.py` file.
``` bash
cd methods/DCE/
python main.py --train True
```

### Compare methods
Then, we can compare the 2 methods described above, and a naive gamma correction.
The script will save the images enhanced under a `results/` folder, and output a comparision about: latencies, and several FIQA metrics (PSNR, MSE, MAE and SSIM).

**N.B.** You need to train the Zero-DCE model before.
``` bash
python main.py --image 2.png
```