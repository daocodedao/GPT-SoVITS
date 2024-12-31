import re

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

inStr="%"
print(is_all_chinese_or_english_punctuation(inStr))  # 应该返回 True
