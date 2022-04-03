# Baseline_FGSBIR
### Baseline Triplet Loss Based Model for Fine-Grained Sketch Based Image Retrieval.

## Pre-trained Models for Inference

Download and unzip models file in `Baseline_FGSBIR`

* [Pre-trained Models](https://mega.nz/file/IgBVDQrD#qxdB2hNazSTbV1_QdQwO2AamWveCsBTk3AGieZ8jmDQ)

## Dependencies

Simply run the following commands:

```bash
conda create --channel conda-forge --name airobj
conda activate airobj
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install pyyaml opencv-python scipy tqdm pycocotools
cd ./AirObject/cocoapi/PythonAPI
python setup.py build
python setup.py install
```

## Get Airobj descriptor


First edit the 3 yaml files in the `config` directory. Next, edit and run the below file

```
python Code/airobj_inference.py
```

## Citation

Using ideas from the below paper.

```txt
@inproceedings{keetha2022airobject,
  title     = {AirObject: A Temporally Evolving Graph Embedding for Object Identification},
  author    = {Keetha, Nikhil Varma and Wang, Chen and Qiu, Yuheng and Xu, Kuan and Scherer, Sebastian}, 
  booktitle = {CVPR},
  year      = {2022},
  url       = {https://arxiv.org/abs/2111.15150}}
```

