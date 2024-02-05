import requests
from utils.util import Util
from urllib.parse import urlencode,unquote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 请求
class Curl:

  # GET、POST、PUT、HEAD、DELETE
  def Request(url: str, data: str='', method: str='GET', header: dict={}, resType: str='json'):
    # 请求头
    requests.DEFAULT_RETRIES = 10
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)


    param = Util.ArrayMerge({
      'Content-Type': 'application/json; charset=utf-8',  #JSON方式
      'Connection':'close'
    }, header)
    # 发送
    if method=='GET' : text = session.get(url, data=data, headers=param).text
    elif method=='POST' : text = session.post(url, data=data, headers=param).text
    elif method=='PUT' : text = session.put(url, data=data, headers=param).text
    elif method=='HEAD' : text = session.head(url, data=data, headers=param).text
    elif method=='DELETE' : text = session.delete(url, data=data, headers=param).text
    # 结果
    if resType=='json' : res=Util.JsonDecode(text)
    else: res=text
    return res

  # URL参数-生成
  def UrlEncode(data: dict):
    return urlencode(data)
  # URL参数-解析
  def UrlDecode(data: str):
    res = {}
    arr = data.split('&')
    for v in arr :
      tmp = v.split('=')
      if len(tmp)==2 : res[tmp[0]] = unquote(tmp[1])
    return res
