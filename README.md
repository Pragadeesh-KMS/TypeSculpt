# TypeSculpt: Text-to-3D with Personalized Precision using Adaptive Attention Mechanism

[![Type Scuplt](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UGOIbAyYS7H1WxjuZCPzgumAPs_9C-YQ?usp=sharing) | [![Type Scuplt](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/kaybe02/type-sculpt-demo)

[Colab Notebook](https://colab.research.google.com/drive/1UGOIbAyYS7H1WxjuZCPzgumAPs_9C-YQ?usp=sharing) | [Kaggle Notebook](https://www.kaggle.com/code/kaybe02/type-sculpt-demo)

TypeSculpt is a cutting-edge text-to-3D generation system that employs an adaptive attention mechanism to create highly personalized and detailed 3D objects from textual descriptions. It leverages state-of-the-art deep learning techniques to bridge the gap between natural language and 3D representation, enabling users to effortlessly generate 3D models tailored to their specific requirements.

## Installation

1. **Python and Environment Setup**

   We recommend using Python >=3.10, PyTorch >=2.1.0, and CUDA >=12.1. Follow these steps to set up your environment:

   - Install Python 3.10 or later from the official [Python website](https://www.python.org/downloads/).
   - Create a new virtual environment and activate it:

     ```bash
     python -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     ```

   - Install PyTorch and CUDA following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

2. **Clone the Repository**

   ```bash
   git clone -b dev https://github.com/camenduru/InstantMesh
   cd InstantMesh
   ```

3. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Text-to-Image Generation**

   ```bash
   python get_images.py
   ```

2. **Image-to-Multiview Generation**

   ```bash
   python get_mv.py
   ```

3. **Multiview-to-3D Generation**

   ```bash
   python get_3d.py
   ```

## Authors

- Pragadeesh KMS [kmspragadeesh6000@gmail.com]
- Koushik Babu VS [koushikbabu90@gmail.com]
- Adithya G [adi442002@gmail.com]

## Acknowledgement

```bibtex
@article{xu2024instantmesh,
 title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
 author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
 journal={arXiv preprint arXiv:2404.07191},
 year={2024}
}
```


We would like to acknowledge the following projects and resources for their valuable contributions:

- Zero123+
- One-2-3-45++
- Instant3D
- Instant NGP
