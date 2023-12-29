### 1. Usage
+ Prepare the data:
    - Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/)
    - [BCDD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
    - [SYSU](https://github.com/liumency/SYSU-CD)
    - Crop LEVIR and BCDD datasets into 256x256 patches. 
    - Generate list file as `ls -R ./label/* > test.txt`
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n RCTNet python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `
    - `cd rctnet/tools`
+ Evaluate pretrained models
    If you want to evaluate your trained model, you can run:
    - `sh test.sh`
+ Train your model
    You can re-train our modelby using:
    - `sh train.sh`


