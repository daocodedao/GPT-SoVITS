#!/bin/bash

workdir=/data/work/GPT-SoVITS/
cd $workdir

. colors.sh

venvBinDir=venv/bin/
pythonPath=${workdir}${venvBinDir}python
echo "Python path:  $pythonPath"

# echo "${YELLOW}source venv_sovits/bin/activate${NOCOLOR}"
# source venv_sovits/bin/activate

helpFunction()
{
   echo ""
   echo "Usage: $0 -m mode -s sourcePath -i processId -ov outVocalPath -oi outInsPath"
   echo -e "\t-s sourcePath"
   echo -e "\t-i processId"
   echo -e "\t-ov outVocalPath"
   echo -e "\t-oi outInsPath"
   exit 1 # Exit script after printing help
}


jobName=utilUvr.py 
echo "${YELLOW}check $jobName pid${NOCOLOR}"
echo "ps aux | grep "$jobName" | grep -v grep  | awk '{print $2}'"
TAILPID=`ps aux | grep "$jobName" | grep -v grep | awk '{print $2}'`  
if [[ "0$TAILPID" != "0" ]]; then
echo "${RED}kill process $TAILPID${NOCOLOR}"
sudo kill -9 $TAILPID
fi


while getopts "s:i:ov:oi:" opt
do
   case "$opt" in
      s ) sourcePath="$OPTARG" ;;
      i ) processId="$OPTARG" ;;
      ov ) outVocalPath="$OPTARG" ;;
      oi ) outInsPath="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

[[ -z  $sourcePath ]] &&  echo -e "${RED}sourcePath is empty ${NOCOLOR}" &&  exit 1
[[ -z  $processId ]] &&  processId=""
[[ -z  $outVocalPath ]] &&  outVocalPath=""
[[ -z  $outInsPath ]] &&  outInsPath=""


echo -e "${YELLOW}${pythonPath} $jobName  -s \"$sourcePath\"  -i \"$processId\" -ov \"$outVocalPath\" -oi \"$outInsPath\" ${NOCOLOR}"
# python -s  /data/work/translate/eR4G4khR6r8/eR4G4khR6r8.mp4 -i eR4G4khR6r8 -oi /data/work/translate/eR4G4khR6r8/eR4G4khR6r8-ins.wav
${pythonPath} $jobName  -s "$sourcePath" -i "$processId" -oi "$processId" -ov "$outVocalPath"

