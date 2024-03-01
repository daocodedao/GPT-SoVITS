import json
import os

def findRoleContent(jsonPath="role.json",  roleName="he"):
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as inFile:
            dataList = json.load(inFile)
            for data in dataList:
                if data["name"] == roleName:
                    return data
    return None


def getAllRole(jsonPath="role.json"):
    roleList = []
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as inFile:
            dataList = json.load(inFile)
            for data in dataList:
                roleName = data["name"] + data["gender"]
                roleList.append(roleName)
    return roleList