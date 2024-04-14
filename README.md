# T2LR-Net
The official TensorFlow implementation of **T2LR-Net**.

Two published paper are related to this code,

> [1] **Zhang, Yinghao**, Peng Li, and Yue Hu. "T2LR-Net: An unrolling network learning transformed tensor low-rank prior for dynamic MR image reconstruction." *Computers in Biology and Medicine* 170 (2024): 108034. ([Journal Paper](https://www.sciencedirect.com/science/article/pii/S0010482524001185))
>
> [2] **Zhang, Yinghao**, Peng Li, and Yue Hu. "Dynamic Mri Using Learned Transform-Based Tensor Low-Rank Network (LT 2 LR-NET)." *2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)*. IEEE, 2023. ([Conference Paper](https://ieeexplore.ieee.org/abstract/document/10230437/))

The tensor low-rank prior has attracted considerable attention in dynamic MR reconstruction. Tensor low-rank methods preserve the inherent high-dimensional structure of data, allowing for improved extraction and utilization of intrinsic low-rank characteristics. However, most current methods are still confined to utilizing low-rank structures either in the image domain or predefined transformed domains. Designing an optimal transformation adaptable to dynamic MRI reconstruction through manual efforts is inherently challenging. In this paper, we propose a deep unrolling network that utilizes the convolutional neural network (CNN) to adaptively learn the transformed domain for leveraging tensor low-rank priors. Under the supervised mechanism, the learning of the tensor low-rank domain is directly guided by the reconstruction accuracy. Specifically, we generalize the traditional t-SVD to a transformed version based on arbitrary high-dimensional unitary transformations and introduce a novel unitary transformed tensor nuclear norm (UTNN). Subsequently, we present a dynamic MRI reconstruction model based on UTNN and devise an efficient iterative optimization algorithm using ADMM, which is finally unfolded into the proposed T2LR-Net. Experiments on two dynamic cardiac MRI datasets demonstrate that T2LR-Net outperforms the state-of-the-art optimization-based and unrolling network-based methods.

![aa](https://yhao-img-bed.obs.cn-north-4.myhuaweicloud.com/202404142029788.png)

## 1. Environment Configuration

- we recommend to use docker

  ```shell
  # pull the docker images
  docker pull yhaoz/tf:2.9.0-bart
  # then you can create a container to run the code, see docker documents for more details
  ```

- if you don't have docker, you can still configure it via installing the requirements by yourself

  ```shell
  pip install -r requirements.txt # tensorflow is gpu version
  ```

Note that, we only run the code in NVIDIA GPU. In our implementation, the code can run normally in both Linux & Windows system.

## 2. Data

We only provide the OCMR test dataset and the corresponding radial-16 under-sampling masks as demo data here. You may get the corresponding files in ***Release*** page of this repo. The files are zipped into `data.zip`. You should download, put and unzip it into the `./data` file folder.

At the time we train the T2LR-Net, publicly available data for dynamic MRI is very few. We could only use a small-size dataset. However, in the year we've been writing and revising, a large number of dynamic MRI raw data were open, such as [OCMR](https://www.ocmr.info/) and [CMRxRecon](https://cmrxrecon.github.io/Home.html). Thus, we highly recommend you to use a large-size dataset to fully investigate your to-be-proposed network.

We have released a repo for processing the OCMR raw data as `.tfrecord` dataset file for TensorFlow, see [yhao-z/ocmr-preproc-tf](https://github.com/yhao-z/ocmr-preproc-tf). If interested, you could refer to it to get more details about generating a ready-to-use dataset. Note that we did not use this repo in our T2LR-Net implementation. If you use this repo, you may obtained different reconstruction results from the published paper.

## 3. Run the code

### Test only

We provide the training weights of our T2LR-Net for radial-16 sampling cases in OCMR dataset. Note that the provided weights are only applicable in our provided dataset with radial-16 sampling. **If you are using other different configuration, retraining from scratch is highly needed.**

```shell
# testing
python test.py
```

### Training

You may need to generate a `.tfrecord` training dataset first.

```shell
# Please refer to main.py for more configurations.
python main.py
```

## 4. Citation

If you find this work useful for your research, please cite:

> ```
> @article{zhang2024t2lr,
>   title={T2LR-Net: An unrolling network learning transformed tensor low-rank prior for dynamic MR image reconstruction},
>   author={Zhang, Yinghao and Li, Peng and Hu, Yue},
>   journal={Computers in Biology and Medicine},
>   volume={170},
>   pages={108034},
>   year={2024},
>   publisher={Elsevier}
> }
> 
> @inproceedings{zhang2023dynamic,
>   title={Dynamic Mri Using Learned Transform-Based Tensor Low-Rank Network (LT 2 LR-NET)},
>   author={Zhang, Yinghao and Li, Peng and Hu, Yue},
>   booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
>   pages={1--4},
>   year={2023},
>   organization={IEEE}
> }
> ```



























