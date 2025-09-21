English Documentation Please Click [here](./MD/LOCAL_EN.md)
# 本地训练
### 搭建环境
0. 确保已安装 `Python==3.10`, CMake & C/C++ 编译器, ffmpeg; 
1. 克隆此仓库;
2. 执行 `pip install -r requirements.txt`;
3. 安装相应GPU版本的PyTorch:(python虚拟机会建立相应的cuda版本)具体可参考[PyTorch官方](https://pytorch.org/get-started/previous-versions/)给出的版本对应关系
    ```
   # CUDA 11.6
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    # CUDA 11.7
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    # CUDA 11.8
    pip install torch==2.3.1+cu118  torchvision==0.18.1+cu118 torchaudio==2.3.1  --extra-index-url https://download.pytorch.org/whl/cu118
    # CUDA 12.4
    pip install torch==2.5.0+cu124  torchvision==0.20.0+cu124 torchaudio==2.5.0 --extra-index-url  https://download.pytorch.org/whl/cu124
    # CUDA 12.6
    pip install torch==2.6.0+cu126  torchvision==0.21.0+cu126 torchaudio==2.6.0 --extra-index-url  https://download.pytorch.org/whl/cu126
   ```
4. 安装处理视频数据必要的库:
    ```
   pip install imageio==2.4.1
   pip install moviepy
   ```
5. 构建monotonic align (训练所需)
    ```
    cd monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    cd ..
    ```
6. 下载训练辅助数据
    ```
    mkdir pretrained_models
    # download data for fine-tuning
    wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
    unzip sampled_audio4ft_v2.zip
    # create necessary directories
    mkdir video_data
    mkdir raw_audio
    mkdir denoised_audio
    mkdir custom_character_voice
    mkdir segmented_character_voice
   ```
7. 下载预训练模型，可用选项有:
    ```
   CJE: Trilingual (Chinese, Japanese, English)
   CJ: Dualigual (Chinese, Japanese)
   C: Chinese only
   ```
   ### Linux
   要下载 `CJE` 模型，请执行以下命令:
    ```
   wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
   wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
   wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json
   ```
   要下载 `CJ` 模型，请执行以下命令:
   ```
   wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth -O ./pretrained_models/D_0.pth
   wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth -O ./pretrained_models/G_0.pth
   wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/config.json -O ./configs/finetune_speaker.json
   ```
    要下载 `C` 模型，请执行以下命令:
   ```
   wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/D_0.pth -O ./pretrained_models/D_0.pth
   wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/G_0.pth -O ./pretrained_models/G_0.pth
   wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/config.json -O ./configs/finetune_speaker.json
   ```
    ### Windows
    从上面描述的三个选项中选择一个，使用URL手动下载`G_0.pth`, `D_0.pth`, `finetune_speaker.json` .
   
    将所有 `G` 模型重命名为 `G_0.pth`, `D`模型重命名为`D_0.pth`, 配置文件(`.json`)重命名为`finetune_speaker.json`.  
    将`G_0.pth`, `D_0.pth` 放在 `pretrained_models` 目录中;  
    将`finetune_speaker.json` 放在 `configs` 目录中;  
   
    #### 请注意，当您下载其中一个时，先前的模型将被覆盖.
8. 将语音数据放在相应的目录下, 详细见 [DATA.MD](./DATA_ZH.MD)， 以了解不同的上传选项.
   ### 短音频
   1. 按照 [DATA.MD](./DATA_ZH.MD) 将数据打包为 `.zip` 压缩文件;  
   2. 将文件放到 `./custom_character_voice/` 目录;
   3. 执行 `unzip ./custom_character_voice/custom_character_voice.zip -d ./custom_character_voice/`
   
   ### 长音频
   4. 根据 [DATA.MD](./DATA_ZH.MD) 重命名音频文件;
   5. 将重命名的音频文件放到 `./raw_audio/` 目录下;
   
   ### 视频
   6. 根据 [DATA.MD](./DATA_ZH.MD) 重命名视频文件;
   7. 将重命名的视频文件放到 `./video_data/` 目录下;
9.  处理所有音频数据.
   ```
   python scripts/video2audio.py
   python scripts/denoise_audio.py
   python scripts/long_audio_transcribe.py --languages "{PRETRAINED_MODEL}" --whisper_size large
   python scripts/short_audio_transcribe.py --languages "{PRETRAINED_MODEL}" --whisper_size large
   python scripts/resample.py
   ```
   根据之前的模型选择替换 `"{PRETRAINED_MODEL}"` 为 `{CJ, CJE, C}` 之一.
   请确保显存至少为12GB。如果不是，请将参数 `whisper_size` 修改为 `medium` 或者 `small`.

10. 处理所有文本数据. 
   如果选择添加辅助数据, 请执行 `python training/preprocess.py --add_auxiliary_data True --languages "{PRETRAINED_MODEL}"`  
   如果没有, 执行 `python training/preprocess.py --languages "{PRETRAINED_MODEL}"`  
    `"{PRETRAINED_MODEL}"` 为 `{CJ, CJE, C}` 之一.
11. 开始训练.  
   执行 `python training/finetune_speaker.py -m ./OUTPUT_MODEL --max_epochs "{Maximum_epochs}" --drop_speaker_embed True`  
    `{Maximum_epochs}` 为想要的epoch数。建议100以上.
   要在之前的检查点继续训练, 请将命令更改为: `python training/finetune_speaker.py -m ./OUTPUT_MODEL --max_epochs "{Maximum_epochs}" --drop_speaker_embed False --cont True`. 在执行此操作之前, 请确保 `./OUTPUT_MODEL/` 目录下有`G_latest.pth` 和 `D_latest.pth`.  
   要查看训练进度, 请打开一个新的终端并 `cd` 到项目目录, 执行 `tensorboard --logdir=./OUTPUT_MODEL`, 然后使用浏览器打开 `localhost:6006` .

12. 训练完成后，可以通过运行以下命令来使用您的模型:  
   `python VITS.py --model_dir ./OUTPUT_MODEL/G_latest.pth --config_dir ./OUTPUT_MODEL/config.json --share True`
13. 要清除所有音频数据，请运行:  
    ### Linux
    ```
    rm -rf ./custom_character_voice/* ./video_data/* ./raw_audio/* ./denoised_audio/* ./segmented_character_voice/* ./separated/* long_character_anno.txt short_character_anno.txt
    ```
    ### Windows
    ```
    del /Q /S .\custom_character_voice\* .\video_data\* .\raw_audio\* .\denoised_audio\* .\segmented_character_voice\* .\separated\* long_character_anno.txt short_character_anno.txt
    ```
## 注意!!!!
### 所有音频应调整为单音频
1.使用CUDA 12.X版本需将 training/finetune_speaker.py 第69行中的`init_method='env://'`改为`init_method='env://?use_libuv=False'`
2.windwos可能需要安装visual studio 2019或者2022并配置cmake
