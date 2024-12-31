
from utility import rolejson as RoleJson
from utility.logger_settings import api_logger


roleDic = RoleJson.findRoleContent(roleName="michael")

api_logger.info(f"找到角色 role {roleDic}")
refer_path = roleDic["refer_path"]
refer_text = roleDic["refer_text"]
refer_language = roleDic["refer_language"]
sovits_path = roleDic["sovits_path"]
gpt_path = roleDic["gpt_path"]