# 说明

原项目 
git clone https://github.com/RVC-Boss/GPT-SoVITS.git

# 安装
## 环境 
```
python3.10 -m venv venv_sovits
source venv_sovits/bin/activate

# cuda11.8
# pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118



pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install modelscope torchaudio sentencepiece funasr


sudo apt install ffmpeg
sudo apt install libsox-dev


```

## 下载模型
```

cd tools/damo_asr/models
# https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files
git clone https://www.modelscope.cn/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
# https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files
git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
# https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files
git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git


# https://huggingface.co/lj1995/GPT-SoVITS/tree/main?clone=true
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/lj1995/GPT-SoVITS

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

cd tools/uvr5/uvr5_weights
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/lj1995/VoiceConversionWebUI



```
#### Install Requirements
```bash
pip install -r requirements.txt
pip uninstall torch torchaudio
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```





# 其他
## 77vpn
```
git config --global http.proxy "http://192.168.0.77:18808"
git config --global https.proxy "http://192.168.0.77:18808"


git config --global --unset https.proxy
git config --global --unset http.proxy

```

## 对外端口
```
cat /data/work/frp/frpc.ini 
vim /data/work/frp/frpc.ini 

[ssh-sovits-9874-web]
type = tcp
local_ip = 127.0.0.1
local_port = 9874
remote_port = 9874
use_encryption = false
use_compression = false

[ssh-sovits-9873-uvr5]
type = tcp
local_ip = 127.0.0.1
local_port = 9873
remote_port = 9873
use_encryption = false
use_compression = false

[ssh-sovits-9872-infer-tts]
type = tcp
local_ip = 127.0.0.1
local_port = 9872
remote_port = 9872
use_encryption = false
use_compression = false

[ssh-sovits-9871-subfix]
type = tcp
local_ip = 127.0.0.1
local_port = 9871
remote_port = 9871
use_encryption = false
use_compression = false

[ssh-sovits-9871-api]
type = tcp
local_ip = 127.0.0.1
local_port = 9880
remote_port = 9880
use_encryption = false
use_compression = false




# 重启frp
sudo systemctl restart  supervisor
sudo supervisorctl reload


sudo supervisord -c /etc/supervisor/supervisord.conf

```

## 查看网速  
```
nethogs
```


## 查看端口
netstat -an | grep 9872




# 资源
## 拷贝
```
scp -r -P 10080 /Users/linzhiji/Documents/code/GPT-SoVITS/resource/ fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS


scp -r -P 10080 /Users/linzhiji/Documents/code/GPT-SoVITS/tools/uvr5/uvr5_weights/ fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/tools/uvr5/

```

## 权限
```
sudo mkdir -p /run/fxbox/
sudo chown -R fxbox:fxbox /run/fxbox/
```


# 操作

## 第一步

伴奏人声分离&去混响&去回声
/data/work/GPT-SoVITS/resource/michael/input


output

音频自动切分输入路径，可文件可文件夹
/data/work/GPT-SoVITS/resource/michael/vocal


批量ASR(中文only)输入文件夹路径
/data/work/GPT-SoVITS/output/slicer_opt


打标数据标注文件路径
/data/work/GPT-SoVITS/output/asr_opt/slicer_opt.list



# 第二步

*实验/模型名
michael

*文本标注文件
/data/work/GPT-SoVITS/output/asr_opt/slicer_opt.list

*训练集音频文件目录
/data/work/GPT-SoVITS/output/slicer_opt
