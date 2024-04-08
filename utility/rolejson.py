import json
import os


def findRoleContent(roleName="he"):
    jsonPath = "/data/work/GPT-SoVITS/role.json"
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as inFile:
            dataList = json.load(inFile)
            for data in dataList:
                if data["name"] == roleName:
                    return data
    return None


def getAllRole():
    jsonPath = "/data/work/GPT-SoVITS/role.json"
    roleList = []
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as inFile:
            dataList = json.load(inFile)
            for data in dataList:
                roleName = data["name"] + data["gender"]
                roleList.append(roleName)
    return roleList