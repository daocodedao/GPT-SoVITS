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


while getopts "p:l:i:s:r:o:" opt
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

cmd="${pythonPath} $workdir$jobName "

[[ -n  $processId ]] && cmd="${cmd} -id $processId"
[[ -n  $language ]] && cmd="${cmd} -tl $language "
[[ -n  $srtPath ]] && cmd="${cmd} -srt $srtPath "
[[ -n  $role ]] && cmd="${cmd} -r $role "
[[ -n  $outPath ]] && cmd="${cmd} -op $outPath "
[[ -n  $prompt ]] && cmd="${cmd} -tp $prompt"


echo "${YELLOW}${cmd}${NOCOLOR}"
${cmd}

