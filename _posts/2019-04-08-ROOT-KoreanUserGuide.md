---
title: "ROOT: Tutorial"
date: 2019-04-08 13:50:00 +0900
categories: ROOT
---

ROOT Data Analysis FrameWork
https://root.cern.ch/root/htmldoc/guides/users-guide/ROOTUsersGuide.html

의 ROOT Users Guide 한국어판,    
ROOT를 사용할 때 알아두면 좋은 명령어들과 코드들을 정리했습니다.

### Install
ROOT 설치: <https://root.cern.ch>    
ROOT가 설치된 경로 찾기 및 버전 확인
```
$echo $ROOTSYS
```

### Execute
터미널에서 루트를 실행하는 방법
```
#in the workstation,
$root # or root -l
root [0]
root [0].q #root 끄기
$
```
root -l 옵션을 붙이면 루트 부팅 메세지 및 로고가 안나옵니다.    
터미널에서 root 명령어를 사용하면 root가 실행되고,
root [0] 상태로 들어갑니다.
root []상태에서 .q 를 입력하면 root를 종료할 수 있습니다.


roofile을 여는 방법
```
TFile f("filename.root")
```
이 때부터 TFile function을 이용하여, 여러가지 방법으로 루트를 가지고 놀 수 있습니다.    
예제로 nanoAOD_3.root 파일을 열어보겠습니다.
```
root []f.ls()
TFile**		nanoAOD_3.root
 TFile*		nanoAOD_3.root
  OBJ: TTree	event	event : 0 at: 0x2838b40
  KEY: TTree	event;1	event
  KEY: TH1D	nevents;1	nevents
  KEY: TH1D	genweight;1	genweight
  KEY: TH1D	weight;1	weight
  KEY: TH1D	cutflow;1	cutflow
```

여기서 
```
t = f.Get("treename")
t.Scan() # Scan("변수1") 이용하면 특정하게 보기
```
을 이용하여 tree별로 데이터 구조를 확인할 수도 있고,

```
treename->Show() #Show("eventnumber") 특정하게 보기
treename->Print()
treename->Scan()
treename->Scan("변수1:변수2:변수3:...")
```
으로 내용을 자세하게 살펴볼 수 있습니다.


### Draw
변수별로 plot을 그리는 방법은
```
treename->Draw("변수1")
treename->Draw("변수1:변수2")
treenam->Draw("변수1", "변수2>0&&변수7<8")
```
이와 같고, condition을 넣어서 그릴 수도 있습니다.

### Merge, hadd
Merge the files, 파일 합치기
```
# ex) hadd -f 결과물이름.파일형식 합치려는파일1.파일형식 합치려는파일2.파일형식 합치려는파일3.파일형식
hadd -f combinedResult.root file1.root file2.root file3.root
```
100GB 이상일 때에는 약간의 코드가 더 필요합니다.    

combineFiles.py 
```
import ROOT
import os, sys

print 'Merging %s' % sys.argv[1]

print "Max tree size",ROOT.TTree.GetMaxTreeSize()
ROOT.TTree.SetMaxTreeSize(200000000000) # 200 Gb
print "Updated tree size",ROOT.TTree.GetMaxTreeSize()

rootMerge = ROOT.TFileMerger(False)
rootMerge.SetFastMethod(True)

path = '/path/to/files/'
file_output = '%s.root' % sys.argv[1]
file_list = ["file1.root", "file2.root", "file3.root", "file4.root"]
for path, dirs, files in os.walk(path):
  for filename in files:
    if ('%s_part' % sys.argv[1]) in filename: file_list.append(path+filename)

print "Input file list:",file_list
print "Output file:",file_output

for file in file_list:

    print "Adding ->", file
    rootMerge.AddFile(file)

rootMerge.OutputFile(file_output)
rootMerge.Merge()
```
