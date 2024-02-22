#!/bin/bash

workdir=/data/work/GPT-SoVITS/
cd $workdir

. colors.sh


echo "${YELLOW}source venv_sovits/bin/activate${NOCOLOR}"
source venv_sovits/bin/activate

helpFunction()
{
   echo ""
   echo "Usage: $0 -m mode -p prompt -id processId -m modelName"
   echo -e "\t-p prompt"
   echo -e "\t-l language"
   echo -e "\t-i processId"
   echo -e "\t-s srt file path"
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


while getopts "p:l:i:s:" opt
do
   case "$opt" in
      p ) prompt="$OPTARG" ;;
      l ) language="$OPTARG" ;;
      i ) processId="$OPTARG" ;;
      s ) srtPath="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

[[ -z  $prompt ]] &&  echo -e "${RED}prompt is empty ${NOCOLOR}" &&  exit 1
[[ -z  $processId ]] &&  processId=""
[[ -z  $language ]] && language="zh"
[[ -z  $srtPath ]] && srtPath=""

echo -e "${YELLOW}python3 $jobName  -tp \"$prompt\"   -tl \"$language\" -id \"$processId\" -srt \"$srtPath\" ${NOCOLOR}"
python3 $jobName  -tp "$prompt" -tl "$language" -id "$processId" -srt "$srtPath"

