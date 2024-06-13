import time
import datetime
import json
import os
from dateutil.relativedelta import relativedelta

import platform
from PIL import Image, ImageOps
import random
import subprocess
from utility.logger_settings import api_logger
import re
from urllib.parse import urlparse


def split(todo_text):
    splits = {"，", "。", "？", "！", ",", ".",
              "?", "!", "~", ":", "：", "—", "…", }
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

# 常用工具


class Utility:

    # 执行Linux命令
    def Exec(cmd: str):
        res = os.popen(cmd)
        return res.readlines()

    # 格式化时间
    def Date(format: str = '%Y-%m-%d %H:%M:%S', timestamp: float = None):
        t = time.localtime(timestamp)
        return time.strftime(format, t)

    def DateFormat(format: str = '%Y-%m-%d %H:%M:%S', duration: str = '0s'):
        l = int(duration[:-1])
        r = duration[-1:]
        # 年、月、周、日、时、分、秒
        now = datetime.datetime.now()
        if r == 'y':
            d = now+relativedelta(years=l)
        elif r == 'm':
            d = now+relativedelta(months=l)
        elif r == 'w':
            d = now+relativedelta(weeks=l)
        elif r == 'd':
            d = now+relativedelta(days=l)
        elif r == 'h':
            d = now+relativedelta(hours=l)
        elif r == 'i':
            d = now+relativedelta(minutes=l)
        elif r == 's':
            d = now+relativedelta(seconds=l)
        else:
            now+relativedelta(seconds=0)
        return d.strftime(format)

    # 时间戳
    def Time():
        return int(time.time())

    # String To Timestamp
    def StrToTime(day: str = None, format: str = '%Y-%m-%d %H:%M:%S'):
        tArr = time.strptime(day, format)
        t = time.mktime(tArr)
        return t if t > 0 else 0

    # Timestamp To GmtIso8601
    def GmtISO8601(timestamp: int):
        t = time.localtime(timestamp)
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", t)

    # 去首尾空格
    def Trim(content, charlist: str = None):
        text = str(content)
        return text.strip(charlist)

    # String to List
    def Explode(delimiter: str, string: str):
        return string.split(delimiter)

    # List to String
    def Implode(glue: str, pieces: list):
        return glue.join(pieces)

    # Array to String
    def JsonEncode(data):
        try:
            return json.dumps(data)
        except Exception as e:
            return ''

    # String to Array
    def JsonDecode(data: str):
        try:
            return json.loads(data)
        except Exception as e:
            return []

    # 合并数组
    def ArrayMerge(*arrays: dict):
        res = {}
        for arr in arrays:
            for k, v in arr.items():
                res[k] = v
        return res

    # Url to Array
    def UrlToArray(url: str):
        if not url:
            return {}
        arr = url.split('?')
        path = arr[1] if len(arr) > 1 else arr[0]
        arr = path.split('&')
        param = {}
        for v in arr:
            tmp = v.split('=')
            param[tmp[0]] = tmp[1]
        return param

    def is_folder(path):
        if os.path.exists(path) and os.path.isdir(path):
            return True
        else:
            return False

    def createFolder(path):
        if os.path.exists(path) and os.path.isdir(path):
            os.makedirs(path)

    def clearDir(path):
        # 删除文件夹中的所有文件
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))

    def isStringInList(srcStr: str, inStrList):
        return any(srcStr in item for item in inStrList)

    def isMac():
        platform_ = platform.system()
        if platform_ == "Mac" or platform_ == "Darwin":
            return True

        return False

    def get_image_paths_from_folder(folder_path):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                for ext in image_extensions:
                    if file.endswith(ext):
                        image_path = os.path.join(root, file)
                        image_paths.append(image_path)

        return image_paths

    def resize_image(image_path, output_path, width, height):
        img = Image.open(image_path)
        saveImg = ImageOps.fit(img, (width, height))
        api_logger.info(f"原始尺寸{img.size}, fit后的尺寸{saveImg.size}")
        saveImg.save(output_path)

    def getMediaDuration(filePath):
        try:
            cmd = f"ffprobe -i {filePath} -show_entries format=duration -v quiet -of csv=\"p=0\""
            result = subprocess.check_output(cmd, shell=True)
            durationFloat = float(result)
            return durationFloat
        except Exception as e:
            return 0

    def getRandomTransitionEffect():
        effects = ["DISSOLVE", "RADIAL", "CIRCLEOPEN", "CIRCLECLOSE", "PIXELIZE", "HLSLICE",
                   "HRSLICE", "VUSLICE", "VDSLICE", "HBLUR", "FADEGRAYS", "FADEBLACK", "FADEWHITE", "RECTCROP",
                   "CIRCLECROP", "WIPELEFT", "WIPERIGHT", "SLIDEDOWN", "SLIDEUP", "SLIDELEFT", "SLIDERIGHT"]
        return random.choice(effects).lower()

    def get_filename_and_extension(s):
        pattern = r'(\w+\.\w+)'
        match = re.search(pattern, s)
        if match:
            return match.group(1)
        else:
            return None

    def sliceStringWithSentence(inStr, sentenceStep=4):
        inStr = inStr.strip("\n")
        inps = split(inStr)
        lenInps = len(inps)
        split_idx = list(range(0, lenInps, sentenceStep))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inStr]
        # return "\n".join(opts)
        return "\n".join(opts)

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
    def isPathAndFileExist(filePath):
        if filePath and os.path.exists(filePath):
            return True
        else:
            return False
        
    def isVideo(filePath):
        video_extensions = [".mp4"]
        for ext in video_extensions:
            if filePath.endswith(ext):
                return True
        return False
    
    def isAudio(filePath):
        audio_extensions = [".mp3",".wav"]
        for ext in audio_extensions:
            if filePath.endswith(ext):
                return True
        return False

    def is_all_chinese_or_english_punctuation(s):
        # 正则表达式匹配中文字符和英文标点符号
        # 中文字符范围：\u4e00-\u9fa5
        # 英文标点符号范围参考：https://www.ascii-code.com/，常见的如[!-/:-@[-`{-~]
        pattern = r'^[\u4e00-\u9fa5!-\/:-@\[-`{-~]*$'
        
        # 使用re.match检查字符串是否完全匹配上述正则表达式
        if re.match(pattern, s):
            return True
        else:
            return False