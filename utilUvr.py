from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
import os
import traceback
import librosa,ffmpeg
import torch
from utility.logger_settings import api_logger
import subprocess
import shutil
import argparse
from utility.utility import Utility
import time
# model_name 模型名
# srcDir 输入待处理音频文件夹路径
# srcPaths 也可批量输入音频文件, 二选一, 优先读文件夹 
# save_root_vocal 指定输出主人声文件夹
# save_root_ins 指定输出非主人声文件夹
# agg 人声提取激进程度，0-20，默认10
# format0 导出文件格式

# scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/translate/eR4G4khR6r8/eR4G4khR6r8.wav /Users/linzhiji/Downloads/eR4G4khR6r8/
# scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/translate/eR4G4khR6r8/eR4G4khR6r8.mp4 /Users/linzhiji/Downloads/eR4G4khR6r8/
# scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/translate/eR4G4khR6r8/ins/ /Users/linzhiji/Downloads/eR4G4khR6r8/
# scp -r -P 10069 fxbox@bfrp.fxait.com:/data/work/translate/eR4G4khR6r8/vocal/ /Users/linzhiji/Downloads/eR4G4khR6r8/


def uvr(modelPath, srcFilePath, outVocalDir, outInsDir,  agg=10, outFormat="wav", ):
    api_logger.info(f"使用模型 {modelPath}, 从 {srcFilePath} 剥离音频")
    # infos = []
    is_half = False
    device="cuda"
    outDir = os.path.dirname(srcFilePath)
    outTempDir = os.path.join(outDir, "temp/")
    if outTempDir is not None:
        os.makedirs(outTempDir, exist_ok=True)
        shutil.rmtree(outTempDir)
        os.makedirs(outTempDir, exist_ok=True)

    srcFileName = os.path.basename(srcFilePath)
    srcFilenameWithoutExt = os.path.splitext(os.path.basename(srcFilePath))[0]
        
    if torch.backends.mps.is_available():
        device = "mps"

    model_name = os.path.splitext(os.path.basename(modelPath))[0]
    api_logger.info(f"模型名： {model_name}")
    try:
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=modelPath,
                device=device,
                is_half=is_half,
            )

        api_logger.info(f"outInsDir={outInsDir}  outVocalDir={outVocalDir} outTempDir={outTempDir}")
        if(os.path.isfile(srcFilePath)==False):
            api_logger.error("srcFilePath 不是文件!")
            return
        need_reformat = 1
        done = 0
        api_logger.info( f"pre_fun类型是： {type(pre_fun)}" )

        try:
            info = ffmpeg.probe(srcFilePath, cmd="ffprobe")
            api_logger.info("ffprobe 探测")
            api_logger.info(info)
            api_logger.info(info["streams"][0]["channels"])
            api_logger.info(info["streams"][0]["sample_rate"])
            if (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100"):
                need_reformat = 0
                pre_fun._path_audio_(srcFilePath, outInsDir, outVocalDir, outFormat, is_hp3)
                done = 1
        except:
            api_logger.error("pre_fun._path_audio_ 异常")
            need_reformat = 1
            api_logger.error(traceback.format_exc())

        if need_reformat == 1:
            api_logger.info("需要重新 reformatted")
            # tmp_path = "%s/%s.reformatted.wav" % (
            #     outTempDir,
            #     os.path.basename(srcFilePath),
            # )
            tmp_path = f"{outTempDir}{srcFilenameWithoutExt}-reformatted.wav"
            command = f"ffmpeg -y -i {srcFilePath} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} "
            api_logger.info(command)
            os.system(command)
            srcFilePath = tmp_path

        try:
            if done == 0:
                api_logger.info("需要重新 _path_audio_")
                pre_fun._path_audio_(
                    srcFilePath, outInsDir, outVocalDir, outFormat, is_hp3
                )
            # api_logger.info()
            # infos.append("%s->Success" % (os.path.basename(srcFilePath)))
            # yield "\n".join(infos)
        except:
            api_logger.error("需要重新 _path_audio_ 异常")
            api_logger.error(traceback.format_exc())
            # infos.append(
            #     "%s->%s" % (os.path.basename(srcFilePath), traceback.format_exc())
            # )
            # yield "\n".join(infos)
    except:
        api_logger.error("整体异常")
        api_logger.error(traceback.format_exc())
        # infos.append(traceback.format_exc())
        # yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            api_logger.error("删除模型异常")
            api_logger.error(traceback.format_exc())

        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# model
# HP2_all_vocals.pth  
# HP2-人声vocals+非人声instrumentals.pth  
# HP5_only_main_vocal.pth  
# HP5-主旋律人声vocals+其他instrumentals.pth  
# onnx_dereverb_By_FoxJoy  
# VR-DeEchoAggressive.pth  
# VR-DeEchoDeReverb.pth  
# VR-DeEchoNormal.pth


def parse_args() -> None: 
    parser = argparse.ArgumentParser(description="GPT-SoVITS")

    parser.add_argument("-s", "--sourcePath", type=str, help="原视频或者音频地址, 音频格式wav")
    parser.add_argument("-i", "--processId", type=str, help="processId")
    parser.add_argument("-ov", "--outVocalPath", type=str, help="输出人声路径")
    parser.add_argument("-oi", "--outInsPath", type=str, help="输出背景音乐路径")

    args = parser.parse_args()
    return args


args = parse_args()

modelPath = "tools/uvr5/uvr5_weights/HP2_all_vocals.pth"
srcPath = args.sourcePath
if srcPath is None or not os.path.exists(srcPath):
    api_logger.error(f"srcPath 为空, 或者{srcPath}不存在")
    exit(1)

processId = args.processId
if len(processId) == 0:
    timestamp = time.time()
    processId = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))

if srcPath is None:
    api_logger.error(f"processId 为空")
    exit(1)

outVocalPath = args.outVocalPath
outInsPath = args.outInsPath

if not outVocalPath and not outInsPath:
    api_logger.error(f"outInsPath 且 outVocalPath为空")
    exit(1)

videoDir = os.path.dirname(srcPath)
srcAudioPath = f"{videoDir}/{processId}.wav"

outVocalDir = os.path.join(videoDir, "vocal/")
outInsDir = os.path.join(videoDir, "ins/")
if outVocalDir is not None:
    os.makedirs(outVocalDir, exist_ok=True)
if outInsDir is not None:
    os.makedirs(outInsDir, exist_ok=True)

api_logger.info(f"清空{outInsDir}")
shutil.rmtree(outInsDir)
api_logger.info(f"清空{outVocalDir}")
shutil.rmtree(outVocalDir)

if Utility.isVideo(srcPath):
    api_logger.info("从视频剥离音频文件")
    # ffmpeg -y -i eR4G4khR6r8.mp4 -vn -acodec pcm_f32le -ac 2 -ar 44100 eR4G4khR6r8.wav
    command = f"ffmpeg -y -i {srcPath} -vn -acodec pcm_f32le -ac 2 -ar 44100 {srcAudioPath}"
    api_logger.info(command)
    result = subprocess.check_output(command, shell=True)

if Utility.isAudio(srcAudioPath):
    api_logger.info("原始文件是音频")
    # command = f"ffmpeg -y -i {srcPath} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} "
    command = f"ffmpeg -y -i {srcPath} -vn -acodec pcm_f32le -ac 2 -ar 44100 {srcAudioPath}"
    api_logger.info(command)
    result = subprocess.check_output(command, shell=True)


api_logger.info("准备剥离背景音乐")
uvr(modelPath=modelPath, srcFilePath=srcAudioPath, outVocalDir=outVocalDir, outInsDir=outInsDir)
api_logger.info("done")


if outVocalPath and len(outVocalPath)>0:
    api_logger.info(f"{outVocalPath} 存在，准备提取")
    paths = [os.path.join(outVocalDir, name) for name in os.listdir(outVocalDir)]
    if paths and len(paths) > 0:
        path = paths[0]
        shutil.copy(path, outVocalPath)

if outInsPath and len(outInsPath)>0:
    api_logger.info(f"{outInsPath} 存在，准备提取")
    paths = [os.path.join(outInsDir, name) for name in os.listdir(outInsDir)]
    if paths and len(paths) > 0:
        path = paths[0]
        shutil.copy(path, outInsPath)

api_logger.info("完成音频剥离")
exit(0)
# for insPath in outInsDir:
#     api_logger.info(insPath)