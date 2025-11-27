# 环境准备
本实验使用的主要环境信息如下:

| 依赖软件              | 版本     |
|:------------------|:-------|
| CANN              | 8.2RC1 |
| Python            | 3.10   |
| MindSpore         | 2.7.1  |
| MindSpeed-Core-MS | r0.4.0 |

[dockerfile_unified](./dockerfiles/dockfile_unified)中打入了**CANN**与**Python**, 开发者可基于此镜像或任何包含指定**CANN**和**Python**版本的环境中完成实验。
其他依赖安装参考以下步骤。

## MindSpore安装

安装MindSpore 2.7.1版本，安装教程请参考[MindSpore快速安装](https://www.mindspore.cn/install)。

执行以下命令

```bash

python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"

```
如果输出

```text

MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!

```
说明MindSpore安装成功。

## MindSpeed-Core-MS及相关依赖安装
```bash
# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh llm

```

## 1. 模型下载
```bash
cd MindSpeed-LLM
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./model_from_hf/qwen2.5_7b_hf
```

## 2.数据集下载
```bash
mkdir dataset && cd dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

## 3. 数据预处理

执行命令
```bash
bash examples/mindspore/qwen25/data_convert_qwen25_pretrain.sh
```
## 4. 权重转换
在权重转换脚本`examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh`中配置`tp并行大小--target-tensor-parallel-size`，`pp并行大小--target-pipeline-parallel-size`，`模型加载路径--load-dir`，`模型保存路径--save-dir`，`模型tokenizer路径--tokenizer-model`。   
- tp2pp4参考脚本如下：
```bash
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2.5_7b_hf/ \
       --save-dir ./tp2pp4/ \
       --tokenizer-model ./model_from_hf/qwen2.5_7b_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```

执行命令
```bash
bash examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
```

## 5.预训练
在与训练脚本`examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh`中配置`模型加载路径CKPT_LOAD_DIR`，`模型保存路径CKPT_SAVE_DIR`，`数据集路径DATA_PATH`，`tokenizer路径TOKENIZER_PATH`

- tp2pp4参考脚本如下：
```bash
CKPT_LOAD_DIR="./tp2pp4/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp2pp4"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=2
PP=4
SEQ_LEN=4096

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
"

msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-save-path ./logs/prof \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-with-memory \
    --profile-with-stack \
     --profile-ranks -1 \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --ai-framework mindspore \
    | tee logs/pretrain_qwen25_7b_32k.log
```


执行命令
```bash
bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh
```

## 6. 查看结果 
运行日志保存在`./logs/prof`中   
MindStudio-Insight下载链接：https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann   
MindStudio-Insight使用方法：https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/debug/profiler.html#%E6%A6%82%E8%A7%88%E7%95%8C%E9%9D%A2%E6%80%BB%E8%A7%88%E6%95%B0%E6%8D%AE%E6%83%85%E5%86%B5