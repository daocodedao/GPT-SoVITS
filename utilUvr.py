from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
import os
import traceback
import librosa,ffmpeg
import torch
from utility.logger_settings import api_logger
import subprocess
# model_name 模型名
# srcDir 输入待处理音频文件夹路径
# srcPaths 也可批量输入音频文件, 二选一, 优先读文件夹 
# save_root_vocal 指定输出主人声文件夹
# save_root_ins 指定输出非主人声文件夹
# agg 人声提取激进程度，0-20，默认10
# format0 导出文件格式


def uvr(modelPath, srcFilePath, agg=10, outFormat="wav"):
    api_logger.info(f"使用模型 {modelPath}, 从 {srcFilePath} 剥离音频")
    # infos = []
    is_half = False
    device="cuda"
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

        outDir = os.path.dirname(srcFilePath)
        if(os.path.isfile(srcFilePath)==False):
            api_logger.error("srcFilePath 不是文件!")
            return
        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(srcFilePath, cmd="ffprobe")
            api_logger.info("ffprobe 探测")
            api_logger.info(info)
            if (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100"):
                need_reformat = 0
                pre_fun._path_audio_(srcFilePath, outDir, outDir, outFormat, is_hp3)
                done = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (
                outDir,
                os.path.basename(srcFilePath),
            )
            os.system(
                "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                % (srcFilePath, tmp_path)
            )
            srcFilePath = tmp_path
        try:
            if done == 0:
                pre_fun._path_audio_(
                    srcFilePath, outDir, outDir, outFormat,is_hp3
                )
            # api_logger.info()
            # infos.append("%s->Success" % (os.path.basename(srcFilePath)))
            # yield "\n".join(infos)
        except:
            api_logger.info(traceback.format_exc())
            # infos.append(
            #     "%s->%s" % (os.path.basename(srcFilePath), traceback.format_exc())
            # )
            # yield "\n".join(infos)
    except:
        api_logger.info(traceback.format_exc())
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
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
    # yield "\n".join(infos)


# model
# HP2_all_vocals.pth  
# HP2-人声vocals+非人声instrumentals.pth  
# HP5_only_main_vocal.pth  
# HP5-主旋律人声vocals+其他instrumentals.pth  
# onnx_dereverb_By_FoxJoy  
# VR-DeEchoAggressive.pth  
# VR-DeEchoDeReverb.pth  
# VR-DeEchoNormal.pth

# weight_uvr5_root = "tools/uvr5/uvr5_weights"
# uvr5_names = []
# for name in os.listdir(weight_uvr5_root):
#     if name.endswith(".pth") or "onnx" in name:
#         uvr5_names.append(name.replace(".pth", ""))

modelPath = "tools/uvr5/uvr5_weights/HP2_all_vocals.pth"
processId = "eR4G4khR6r8"
videoPath = f"/data/work/translate/eR4G4khR6r8/{processId}.mp4"
srcAudioPath = f"/data/work/translate/eR4G4khR6r8/{processId}.wav"
videoDir = os.path.dirname(videoPath)

api_logger.info("从视频剥离音频文件")
command = f"ffmpeg -y -i {videoPath} -vn -acodec copy {srcAudioPath}"
result = subprocess.check_output(command, shell=True)

api_logger.info("准备剥离背景音乐")
uvr(modelPath=modelPath, srcFilePath=srcAudioPath)
api_logger.info("done")