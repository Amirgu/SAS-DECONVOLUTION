# SAS-DECONVOLUTION

Impementation of state-of-art Short and sparse blind deconvolution in Python following this paper:

'Short-and-Sparse Deconvolution — A Geometric Approach' by: Lau, Yenson and Qu, Qing and Kuo, Han-Wen and Zhou, Pengcheng and Zhang, Yuqian and Wright, John

### Algorithms Implemented : 

iADM and iADM with homotopy on 1D vectors on : iADM.py

iADM and iADM with homotopy and iADM with reweighting on 2D vectors or images on : iADM2D.py

### Notebooks :

Jupyter notebook of testing recovery of 1D vectors recovery  : test.ipynb

Jupyter notebook of recovering calcium image recovery : Deconvolve2D.ipynb

### Possible Applications :

-The microscopic image of high temperature superconductor polycrystalline Deconvoution: 

![Convolution](https://user-images.githubusercontent.com/127419134/235808446-18ec5884-b01d-4ec9-a9b1-95f2660e24ac.PNG)

-Natural image deblurring (if using image gradient as they are sparse):

![Deblurring](https://user-images.githubusercontent.com/127419134/235808883-d282685b-27f9-4eb9-b5af-14b8d8474e2f.PNG)

-Convolutional Dictionary learning 
### References : 

(https://deconvlab.github.io/)


@article{lau2019short, title={Short-and-Sparse Deconvolution — A Geometric Approach}, author={Lau, Yenson and Qu, Qing and Kuo, Han-Wen and Zhou, Pengcheng and Zhang, Yuqian and Wright, John}, journal={Preprint}, year={2020} } }

@article{kuo2019geometry, title={Geometry and symmetry in short-and-sparse deconvolution}, author={Kuo, Han-Wen and Zhang, Yuqian and Lau, Yenson and Wright, John}, journal={arXiv preprint arXiv:1901.00256}, year={2019} } }







