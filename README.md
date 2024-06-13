## Coherent Event Guided Low-Light Video Enhancement

[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liang_Coherent_Event_Guided_Low-Light_Video_Enhancement_ICCV_2023_paper.html) | [Project Page](https://sherrycattt.github.io/EvLowLight/)


[Jinxiu Liang](https://sherrycattt.github.io/)<sup>1</sup>, [Yixin Yang](https://yixinyang-00.github.io/)<sup>1</sup>, [Boyu Li](https://camera.pku.edu.cn/team)<sup>1</sup>, [Peiqi Duan](https://scholar.google.com/citations?user=VqF8ZNYAAAAJ)<sup>1</sup>, [Yong Xu](https://scholar.google.com/citations?user=1hx5iwEAAAAJ)<sup>2</sup>, [Boxin Shi](https://camera.pku.edu.cn/team)<sup>1</sup>

<sup>1</sup>Peking University<br><sup>2</sup>South China University of Technology

<p align="center">
    <img src="docs/static/images/teaser.jpg">
</p>

---

<p align="center">
    <img src="docs/static/images/method-v8.jpg">
</p>

:star:If EvLowLight is helpful for you, please help star this repo. Thanks!:hugs:

## Table Of Contents

- [TODO](#todo)
- [Installation](#installation)
- [Inference](#inference)
- [Data Preparation](#data)

## <a name="todo"></a>TODO

- [x] Release inference code and pretrained models.
- [x] Update links to paper and project page.
- [x] Provide a runtime environment Docker image.
- [ ] Release train code and training set.

## <a name="installation"></a>Installation
1. Clone this repo using `git`:
    
    ```shell
    git clone https://github.com/sherrycattt/EvLowLight.git
    ```

2. Create environment:

    Option 1: Using [`pip`](https://pip.pypa.io/en/stable/installation/)
    ```shell
    cd EvLowLight
    conda create -n evlowlight python=3.8
    conda activate evlowlight
    pip install -r requirements.txt
    ```
    
    Option 2: Using [`docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    
    ```shell
    docker run --runtime=nvidia --gpus all --ipc=host --network=host  --rm -it \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v `pwd`/EvLowLight:/workspace \
      -v `pwd`/timelens:/datasets/timelens \
      sherrycat/evlowlight
    ```

    Note the installation is only compatible with **Linux** users.

## <a name="inference"></a>Inference

We provide an example for inference, check [options/**_option.yml](options/timelens_option.yml) for more arguments.

```shell
python inference.py -opt options/timelens_option.yml
```

## <a name="data"></a>Data Preparation

We provide example test data converted from the [TimeLens](https://rpg.ifi.uzh.ch/TimeLens.html) for demo, which can be downloaded from [Link](https://disk.pku.edu.cn/link/AA80956FD9F5264BD48EB01951067DE7BE ) (extracted code: Y9CN).
Please place the dataset in the `../datasets` folder. The dataset structure should be organized as follows:

```
├── timelens
│   └── events
│       ├── paprika_1000_gain_control_02
│       │   ├── events.txt
│       │   └── timestamp.txt
│       ├── pen_03
│       │   ├── events.txt
│       │   └── timestamp.txt
│       ...
│   └── low
│       ├── paprika_1000_gain_control_02
│       │   ├── 000000.png
│       │   └── 000001.png
│       │   ...
│       ├── pen_03
│       │   ├── 000000.png
│       │   └── 000001.png
│       │   ...
│       ...
│  ...
```
Each subfolder in the `low` folder contains image files with template filename `%06d.png`, and the file in the `events` subfolder contains events corresponding to the image subfolder with template filename `events.txt` defined as `ev_file_ext` in the [option configuration file](options/timelens_option.yml). 
Moreover, `events` also contains `timestamp.txt` where image timestamps are stored. The image stamps in `timestamp.txt` should match with the image files .

## Citation

Please cite us if our work is useful for your research.

```
@inproceedings{liang2023evlowlight,
  title = {Coherent Event Guided Low-Light Video Enhancement},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  author = {Liang, Jinxiu and Yang, Yixin and Li, Boyu and Duan, Peiqi and Xu, Yong and Shi, Boxin},
  year = {2023},
  pages = {10615--10625},
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.

## Contact

If you have any questions, please feel free to contact with me at `cssherryliang@pku.edu.cn`.