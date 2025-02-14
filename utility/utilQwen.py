
import requests
import socket,os
import json
import re
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from openai import OpenAI

from utility.logger_settings import api_logger

def getNetworkIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]

serverUrl = "http://39.105.194.16:9191/v1/chat/completions/"
kRequestTimeout = 60*2


def composeV1Dic(systemContent, userContent):
    return {
         "model": "qwen2.5:7b-instruct-fp16",
         "messages":[{
            "role": "system",
            "content": systemContent,
            },
            {
            "role":"user",
            "content":userContent
            }]
    }

def normalQwen(messages):
    # print(f"normalQwen request:{messages}")
    response = requests.post(serverUrl, json=messages, timeout=kRequestTimeout)
    # print(f"normalQwen response:{response.json()}")
    if response.status_code == 200:
        retJson = response.json()
        # api_logger.info(retJson)
        retTextList = retJson["choices"]
        ret_text = ""
        if retTextList and len(retJson) > 0:
            ret_text = retTextList[0]["message"]["content"]
        return ret_text
    else:
        api_logger.info("请求失败，状态码：", response.status_code)
        api_logger.info(response.text)
        return ""

def run_gpt(text, system=None):
    if not system:
        systemPrompt = f"""
    你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。回答不要太长。
        """.strip()
    else:
        systemPrompt = system.strip()

    jsonData = composeV1Dic(systemContent=systemPrompt, userContent=text)

    content = normalQwen(messages=jsonData)
    # paragraphLen = len(queryList)
    return content


def run_gpt_withDic(dic):
    systemPrompt = f""