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
git clone https://www.modelscope.cn/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git

cd GPT_SoVITS/pretrained_models
git clone https://huggingface.co/lj1995/GPT-SoVITS


cd tools/uvr5/uvr5_weights
git clone https://huggingface.co/lj1995/VoiceConversionWebUI



```
#### Install Requirements
```bash
pip install -r requirements.txt
pip uninstall torch torchaudio
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```


# 启动

```
source venv_sovits/bin/activate
python webui.py


```

```
# api
source venv_sovits/bin/activate
python api.py -dr "resource/何同学/source.MP3" -dt "在我身后的是10万个纸盒子" -dl "zh" 

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
scp -r -P 10068 /Users/linzhiji/Documents/code/GPT-SoVITS/resource/ fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS


scp -r -P 10068 fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/resource/out/ /Users/linzhiji/Documents/code/GPT-SoVITS/resource/out/

scp -r -P 10080 /Users/linzhiji/Documents/code/GPT-SoVITS/tools/uvr5/uvr5_weights/ fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/tools/uvr5/


scp -r -P 10068 /Users/linzhiji/Documents/code/GPT-SoVITS/resource/FaTiaoZhang/fatiaozhang_e24_s264.pth fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/resource/FaTiaoZhang/

scp -r -P 10068 /Users/linzhiji/Documents/code/GPT-SoVITS/resource/FaTiaoZhang/fatiaozhang-e10.ckpt fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/resource/FaTiaoZhang/

scp -r  /data/work/GPT-SoVITS/resource/FaTiaoZhang/ fxbox@192.168.0.69:/data/work/GPT-SoVITS/resource/

```

## 权限
```
sudo mkdir -p /run/fxbox/
sudo chown -R fxbox:fxbox /run/fxbox/
```


# 操作
操作前先清空 output
## 第一步
http://39.105.194.16:9873/

1.伴奏人声分离&去混响&去回声
算法选 HP5_only
转换成功后，在目录下生成
人声音 output/uvr5_opt_vocal
乐器声音 output/uvr5_opt_ins



2.音频自动切分， 把长音频切割成短音频，方便校对和训练
输入路径 output/uvr5_opt_vocal
输出路径 output/slicer_opt

3.批量ASR(语音识别)
输入文件夹
output/slicer_opt
输出文件夹
output/asr_opt
输出文件
output/asr_opt/slicer_opt.list

4.打标数据标注文件路径
output/asr_opt/slicer_opt.list



# 第二步

1. 格式化数据
*实验/模型名
mich

*文本标注文件
output/asr_opt/slicer_opt.list

*训练集音频文件目录
output/slicer_opt

配置好后，一键三连
2. 微调训练
   训练成功后
   SoVITS_weights/
   GPT_weights/

SoVITS_weights/ 后缀 pth, e代表轮数，s代表步数
GPT_weights/ 后缀 ckpt, e代表轮数
下拉选择模型推理，e代表轮数，s代表步数。不是轮数越高越好。可以试试20, 15

然后上传一段参考音频，建议是数据集中的音频。最好5秒。参考音频很重要！会学习语速和语气，请认真选择。参考音频的文本是参考音频说什么就填什么，必须要填。语种也要对应
接着就是输入要合成的文本了，注意语种要对应。目前可以中英混合，语种选择中文，日英混合，语种选择日文。


scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/SoVITS_weights /Users/linzhiji/Downloads
scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/GPT-SoVITS/GPT_weights /Users/linzhiji/Downloads


# 使用

## 文字生成语音
/data/work/GPT-SoVITS/start-gen-voice-local.sh -l "zh" -i "122324" -p "总的来说，<橙黄橘绿半 甜时> 是一本让人回味无穷，满足感官享受的美文集 。这不仅是一本书，更是一场心灵的盛 宴，一段历史与现在、人文与自然相交的记忆。" 

scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/book/122324/122324.wav /Users/linzhiji/Downloads


## srt生成语音
/data/work/GPT-SoVITS/start-gen-voice-local.sh -l "zh"  -s "/data/work/aishowos/whisper_subtitle/sample/simple5-cn.srt" 