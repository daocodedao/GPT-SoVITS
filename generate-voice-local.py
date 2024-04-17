"""
# handle.py usage
## 执行参数:

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu","mps"`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`

"""
import time
import LangSegment
import config as global_config
from my_utils import load_audio
from module.mel_processing import spectrogram_torch
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import SynthesizerTrn
from io import BytesIO
from feature_extractor import cnhubert
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, HTTPException
import soundfile as sf
import librosa
import torch
from time import time as ttime
import signal
import argparse
import os, re, logging
import sys
from utility.utility import Utility
from utility.logger_settings import api_logger
import platform
import srt
from utility.rolejson import *
from tools.i18n.i18n import I18nAuto
from utility.utility import Utility

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))


g_config = global_config.Config()
g_para = global_config.ParamConfig()
# global g_para
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True

def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True

def get_bert_feature(text, word2ph):
    global g_para
    with torch.no_grad():
        inputs = g_para.tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(g_para.device)  # 输入是long不用管精度问题，精度随bert_model
        res = g_para.bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec

def clean_text_inf(text, language):
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist

def get_bert_inf(phones, word2ph, norm_text, language, device):
    global g_para
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if g_para.is_half == True else torch.float32,
        ).to(device)

    return bert

def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        # print(f"nonen_clean_text_inf clean_text_inf text={textlist[i]} lang={lang}")
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text

def nonen_get_bert_inf(text, language, device):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    # print("nonen_get_bert_inf")
    print(textlist)
    print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        # print(f"nonen_get_bert_inf clean_text_inf text={textlist[i]} lang={lang} device={device}")
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        # print(f"phones={phones}")
        # print(f"word2ph={word2ph}")
        # print(f"norm_text={norm_text}")
        bert = get_bert_inf(phones, word2ph, norm_text, lang, device)
        # print(f"bert={bert}")
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert

def get_cleaned_text_final(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text

def get_bert_final(phones, word2ph, text, language, device):
    if language == "en":
        bert = get_bert_inf(phones, word2ph, text, language, device)
    elif language in {"zh", "ja","auto"}:
        bert = nonen_get_bert_inf(text, language, device)
    elif language == "all_zh":
        bert = get_bert_feature(text, word2ph).to(device)
    else:
        bert = torch.zeros((1024, len(phones))).to(device)
    return bert

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k=5, top_p=1, temperature=1):
    global g_para
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(g_para.hps.data.sampling_rate * 0.3),
                        dtype=np.float16 if g_para.is_half == True else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (g_para.is_half == True):
            wav16k = wav16k.half().to(g_para.device)
            zero_wav_torch = zero_wav_torch.half().to(g_para.device)
        else:
            wav16k = wav16k.to(g_para.device)
            zero_wav_torch = zero_wav_torch.to(g_para.device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = g_para.ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"].transpose(1, 2)  # .float()
        codes = g_para.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    # prompt_language = global_config.dict_language[prompt_language]
    # text_language = global_config.dict_language[text_language]
    prompt_language = prompt_language
    text_language = text_language
    
    phones1, word2ph1, norm_text1 = get_cleaned_text_final(prompt_text, prompt_language)
    bert1=get_bert_final(phones1, word2ph1, norm_text1,prompt_language,g_para.device).to(torch.float16)
    texts = text.split("\n")
    audio_opt = []

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0 or Utility.is_number(text.strip()):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, word2ph2, norm_text2 = get_cleaned_text_final(text, text_language)
        bert2 = get_bert_final(phones2, word2ph2, norm_text2, text_language, g_para.device).to(torch.float16)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1+phones2).to(g_para.device).unsqueeze(0)


        bert = bert.to(g_para.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(g_para.device)
        prompt = prompt_semantic.unsqueeze(0).to(g_para.device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = g_para.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=g_para.hz * g_para.max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(g_para.hps, ref_wav_path)  # .to(device)
        if g_para.is_half == True:
            refer = refer.half().to(g_para.device)
        else:
            refer = refer.to(g_para.device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            g_para.vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(g_para.device).unsqueeze(0), refer
            )
                .detach()
                .cpu()
                .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield g_para.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )

def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def handle_change(path, text, language):
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

    if path != "" or path is not None:
        g_para.default_refer.path = path
    if text != "" or text is not None:
        g_para.default_refer.text = text
    if language != "" or language is not None:
        g_para.default_refer.language = language

    api_logger.info(f"[INFO] 当前默认参考音频路径: {g_para.default_refer.path}")
    api_logger.info(f"[INFO] 当前默认参考音频文本: {g_para.default_refer.text}")
    api_logger.info(f"[INFO] 当前默认参考音频语种: {g_para.default_refer.language}")
    api_logger.info(f"[INFO] is_ready: {g_para.default_refer.is_ready()}")

    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)

def parse_args() -> None: 
    parser = argparse.ArgumentParser(description="GPT-SoVITS")

    parser.add_argument("-dr", "--default-refer-path", type=str,
                        default="resource/he/source.MP3", help="默认参考音频路径")
    parser.add_argument("-dt", "--default-refer-text", type=str,
                        default="在我身后的是10万个纸盒子", help="默认参考音频文本")
    parser.add_argument("-dl", "--default-refer-language",
                        type=str, default="zh", help="默认参考音频语种")
    parser.add_argument("-srt", "--srt-file-path",
                        type=str, default="", help="从srt里读取")
    parser.add_argument("-tp", "--text-prompt", type=str, default="", help="输入文本")
    parser.add_argument("-tl", "--text-language", type=str, default="zh", help="输入文本语言")
    parser.add_argument("-id", "--process-id", type=str, default="", help="process_id")
    parser.add_argument("-r", "--role", type=str, default="FaTiaoZhang", help="role name")

    parser.add_argument("--out-path", type=str)

    args = parser.parse_args()
    return args

def isMac():
    platform_ = platform.system()
    if platform_ == "Mac" or platform_ == "Darwin":
      return True
    
    return False

def initResource():
    api_logger.info("加载模型，加载参数")
    # AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"
    global g_para
    args = parse_args()

    g_para.sovits_path = g_config.pretrained_sovits_path
    g_para.gpt_path = g_config.pretrained_gpt_path
    
    g_para.cnhubert_base_path = g_config.cnhubert_path
    g_para.bert_path = g_config.bert_path
    g_para.is_half = g_config.is_half
    g_para.device = "cuda"
    g_para.srt_path = args.srt_file_path
    g_para.out_path = args.out_path
    if isMac():
        g_para.device = "mps"

    class DefaultRefer:
        def __init__(self, path, text, language):
            self.path = path
            self.text = text
            self.language = language

        def is_ready(self) -> bool:
            return is_full(self.path, self.text, self.language)


    role = args.role
    refer_path = args.default_refer_path
    refer_text = args.default_refer_text
    refer_language = args.default_refer_language
    roleDic = findRoleContent(roleName=role)
    api_logger.info(f"找到角色 role {roleDic}")
    refer_path = roleDic["refer_path"]
    refer_text = roleDic["refer_text"]
    refer_language = roleDic["refer_language"]
    g_para.sovits_path = roleDic["sovits_path"]
    g_para.gpt_path = roleDic["gpt_path"]


    g_para.default_refer = DefaultRefer(refer_path, refer_text, refer_language)

    g_para.text_prompt = args.text_prompt
    g_para.text_language = args.text_language
    # if len(text_language) == 0 or len(text_prompt) == 0:
    #     api_logger.info("错误，没有提示词或提示词语言有误!")
    #     exit(1)
    # else:
    #     text_prompt = Utility.sliceStringWithSentence(text_prompt)

    g_para.process_id = args.process_id
    if len(g_para.process_id) == 0:
        timestamp = time.time()
        g_para.process_id = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))

    # 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
    if g_para.default_refer.path == "" or g_para.default_refer.text == "" or g_para.default_refer.language == "":
        g_para.default_refer.path, g_para.default_refer.text, g_para.default_refer.language = "", "", ""
        api_logger.info("[INFO] 未指定默认参考音频")
    else:
        api_logger.info(f"[INFO] 默认参考音频路径: {g_para.default_refer.path}")
        api_logger.info(f"[INFO] 默认参考音频文本: {g_para.default_refer.text}")
        api_logger.info(f"[INFO] 默认参考音频语种: {g_para.default_refer.language}")

    cnhubert.cnhubert_base_path = g_para.cnhubert_base_path
    g_para.tokenizer = AutoTokenizer.from_pretrained(g_para.bert_path)
    g_para.bert_model = AutoModelForMaskedLM.from_pretrained(g_para.bert_path)
    if g_para.is_half:
        g_para.bert_model = g_para.bert_model.half().to(g_para.device)
    else:
        g_para.bert_model = g_para.bert_model.to(g_para.device)

    # n_semantic = 1024
    dict_s2 = torch.load(g_para.sovits_path, map_location="cpu")
    g_para.hps = dict_s2["config"]

    class DictToAttrRecursive:
        def __init__(self, input_dict):
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    # 如果值是字典，递归调用构造函数
                    setattr(self, key, DictToAttrRecursive(value))
                else:
                    setattr(self, key, value)


    g_para.hps = DictToAttrRecursive(g_para.hps)
    g_para.hps.model.semantic_frame_rate = "25hz"
    api_logger.info(f"准备加载gpt模型: {g_para.gpt_path}")
    dict_s1 = torch.load(g_para.gpt_path, map_location="cpu")
    g_para.config = dict_s1["config"]
    g_para.ssl_model = cnhubert.get_model()
    if g_para.is_half:
        g_para.ssl_model = g_para.ssl_model.half().to(g_para.device)
    else:
        g_para.ssl_model = g_para.ssl_model.to(g_para.device)

    g_para.vq_model = SynthesizerTrn(
        g_para.hps.data.filter_length // 2 + 1,
        g_para.hps.train.segment_size // g_para.hps.data.hop_length,
        n_speakers=g_para.hps.data.n_speakers,
        **g_para.hps.model)
    if g_para.is_half:
        g_para.vq_model = g_para.vq_model.half().to(g_para.device)
    else:
        g_para.vq_model = g_para.vq_model.to(g_para.device)
    g_para.vq_model.eval()
    # api_logger.info(g_para.vq_model.load_state_dict(dict_s2["weight"], strict=False))
    # hz = 50
    g_para.max_sec = g_para.config['data']['max_sec']
    g_para.t2s_model = Text2SemanticLightningModule(g_para.config, "****", is_train=False)
    g_para.t2s_model.load_state_dict(dict_s1["weight"])
    if g_para.is_half:
        g_para.t2s_model = g_para.t2s_model.half()
    g_para.t2s_model = g_para.t2s_model.to(g_para.device)
    g_para.t2s_model.eval()
    total = sum([param.nelement() for param in g_para.t2s_model.parameters()])
    # api_logger.info("Number of parameter: %.2fM" % (total / 1e6))

def handle(inText, 
           text_language, 
           refer_wav_path="", 
           prompt_text="", 
           prompt_language="", 
           output_wav_path=""):
    
    if (refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None):
        refer_wav_path, prompt_text, prompt_language = (
            g_para.default_refer.path,
            g_para.default_refer.text,
            g_para.default_refer.language,
        )
        if not g_para.default_refer.is_ready():
            api_logger.error("未指定参考音频且接口无预设")
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    with torch.no_grad():
        gen = get_tts_wav(refer_wav_path, 
                          prompt_text,
                          prompt_language, 
                          inText, 
                          text_language)
        sampling_rate, audio_data = next(gen)

    # wav = BytesIO()
    # sf.write(wav, audio_data, sampling_rate, format="wav")
    # wav.seek(0)
    if len(output_wav_path) == 0:
        output_dir = f"/data/work/book/{g_para.process_id}/"
        os.makedirs(output_dir, exist_ok=True)
        output_wav_path = os.path.join(output_dir, f"{g_para.process_id}.wav")
    
    # api_logger.info("保存音频到 " + output_wav_path)
    sf.write(output_wav_path, audio_data, sampling_rate)
    torch.cuda.empty_cache()
    if g_para.device == "mps":
        # api_logger.info('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()

    api_logger.info("音频保存到 " + output_wav_path)

    # return StreamingResponse(wav, media_type="audio/wav")

def isIgnore(inStr):
    passWordList = ["此处省略了", "（省略）"]
    for passWord in passWordList:
        if passWord in inStr:
            return True
    return False

initResource()

if g_para.srt_path is not None and os.path.exists(g_para.srt_path) :
    with open(g_para.srt_path, 'r') as srcFile:
        # 读取文件内容
        api_logger.info("读取 srt")
        content = srcFile.read()
        subs = srt.parse(content)

        folder_path = os.path.dirname(g_para.srt_path)
        output_dir = os.path.join(folder_path, f"tts/")
        os.makedirs(output_dir, exist_ok=True)
        Utility.clearDir(output_dir)

        for sub in subs:
            output_wav_path = os.path.join(output_dir, f"{sub.index}.wav")
            # api_logger.info("单字符串转wav")
            api_logger.info(f"准备TTS： {sub.content}")
            if isIgnore(sub.content):
                api_logger.info("跳过，不做TTS")
                continue
            handle(inText=sub.content, 
                   text_language=g_para.text_language, 
                   output_wav_path=output_wav_path)

        api_logger.info(f"处理完成, 输出到文件夹：{output_dir}")

else:
    api_logger.info(f"准备TTS： {g_para.text_prompt}")
    handle(inText=g_para.text_prompt, 
           text_language=g_para.text_language, 
           output_wav_path=g_para.out_path)
