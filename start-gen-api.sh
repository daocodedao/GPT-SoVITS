#!/bin/bash

workdir=/data/work/GPT-SoVITS/
cd $workdir

. colors.sh

venvBinDir=venv_sovits/bin/
pythonPath=${workdir}${venvBinDir}python
echo "Python path:  $pythonPath"

echo "${YELLOW}source ${venvBinDir}activate${NOCOLOR}"
source ${venvBinDir}activate


jobName=api.py 
echo "${YELLOW}check $jobName pid${NOCOLOR}"
echo "ps aux | grep "$jobName" | grep -v grep  | awk '{print $2}'"
TAILPID=`ps aux | grep "$jobName" | grep -v grep | awk '{print $2}'`  
if [[ "0$TAILPID" != "0" ]]; then
echo "${RED}kill process $TAILPID${NOCOLOR}"
sudo kill -9 $TAILPID
fi


# echo -e "${YELLOW}${pythonPath} $jobName -dr \"resource/FaTiaoZhang/source.MP3\" -dt \"这个影展呢是戏影厂跟春光映画联合主办的\" -dl \"zh\" ${NOCOLOR}"
# nohup ${pythonPath} $jobName  -dr "resource/FaTiaoZhang/source.MP3" -dt "这个影展呢是戏影厂跟春光映画联合主办的" -dl "zh"  > /dev/null 2>&1 &


echo -e "${YELLOW}${pythonPath} $jobName  ${NOCOLOR}"
nohup ${pythonPath} $jobName  -dr "resource/FaTiaoZhang/source.MP3"  > /dev/null 2>&1 &




