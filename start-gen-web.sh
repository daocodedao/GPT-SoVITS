#!/bin/bash

workdir=/data/work/GPT-SoVITS/
cd $workdir

. colors.sh

venvBinDir=venv_sovits/bin/
pythonPath=${workdir}${venvBinDir}python
echo "Python path:  $pythonPath"

echo "${YELLOW}source ${venvBinDir}activate${NOCOLOR}"
source ${venvBinDir}activate


jobName=webui.py 
echo "${YELLOW}check $jobName pid${NOCOLOR}"
echo "ps aux | grep "$jobName" | grep -v grep  | awk '{print $2}'"
TAILPID=`ps aux | grep "$jobName" | grep -v grep | awk '{print $2}'`  
if [[ "0$TAILPID" != "0" ]]; then
echo "${RED}kill process $TAILPID${NOCOLOR}"
sudo kill -9 $TAILPID
fi


echo -e "${YELLOW}${pythonPath} $jobName -p 9180 ${NOCOLOR}"
nohup ${pythonPath} $jobName  > logs/webui.log 2>&1 &




