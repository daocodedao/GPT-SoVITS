#!/bin/bash

workdir=/data/work/GPT-SoVITS/
cd $workdir

. colors.sh

venvBinDir=venv_sovits/bin/
pythonPath=${workdir}${venvBinDir}python
echo "Python path:  $pythonPath"

echo "${YELLOW}source ${venvBinDir}activate${NOCOLOR}"
source ${venvBinDir}activate

helpFunction()
{
   echo ""
   echo "Usage: $0 -m mode -p prompt -id processId -m modelName -r role"
   echo -e "\t-p prompt"
   echo -e "\t-l language"
   echo -e "\t-i processId"
   echo -e "\t-s srt file path"
   echo -e "\t-r role"
   exit 1 # Exit script after printing help
}


jobName=generate-voice-local.py 
echo "${YELLOW}check $jobName pid${NOCOLOR}"
echo "ps aux | grep "$jobName" | grep -v grep  | awk '{print $2}'"
TAILPID=`ps aux | grep "$jobName" | grep -v grep | awk '{print $2}'`  
if [[ "0$TAILPID" != "0" ]]; then
echo "${RED}kill process $TAILPID${NOCOLOR}"
sudo kill -9 $TAILPID
fi


while getopts "p:l:i:s:r:o" opt
do
   case "$opt" in
      p ) prompt="$OPTARG" ;;
      l ) language="$OPTARG" ;;
      i ) processId="$OPTARG" ;;
      s ) srtPath="$OPTARG" ;;
      r ) role="$OPTARG" ;;
      o ) outPath="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

[[ -z  $prompt && -z  $srtPath ]] &&  echo -e "${RED}prompt or is empty ${NOCOLOR}" &&  exit 1
[[ -z  $processId ]] &&  processId=""
[[ -z  $language ]] && language="zh"
[[ -z  $srtPath ]] && srtPath=""
[[ -z  $role ]] && role=""
[[ -z  $outPath ]] && outPath=""

echo -e "${YELLOW}${pythonPath} $jobName  -tp \"$prompt\"   -tl \"$language\" -id \"$processId\" -srt \"$srtPath\" -r \"$role\"${NOCOLOR}"
${pythonPath} $jobName  -tl "$language" -id "$processId" -srt "$srtPath" -r "$role" -op "$outPath" -tp "$prompt"

