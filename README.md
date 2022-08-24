# heterochromatin-DAPI
Get features of heterochromatin from DAPI stained nucleus

# install the python virtual environment
download and install anaconda: https://www.anaconda.com/

```
conda create --no-default-packages --name image_py3 python=3.9
conda activate image_py3
conda install -n image_py3 numpy scipy matplotlib h5py seaborn pandas jupyter scikit-image Pillow
conda install -n image_py3 --channel=conda-forge nd2reader imreg_dft opencv

jupyter notebook --notebook-dir=E:\
```

Follow the code in `Get heterochromatin features from DAPI images.ipynb`
