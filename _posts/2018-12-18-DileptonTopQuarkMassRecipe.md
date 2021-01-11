---
title: "Measurement of the top quark mass in dilepton channel"
date: 2018-12-18 14:11:00 +0900
categories: TopQuarkMass
---

CERN의 입자물리 실험 중에서 탑 쿼크의 질량을 측정하기 위한 데이터 분석 방법    

Dongjun Jeong, "Measurement of the top quark mass using charmed meson in b-jet"    
<https://academic.naver.com/article.naver?doc_id=630648558>


Github repository : 
```
CPLUOS           https://github.com/CPLUOS/nano 
Dongjun Jeong    https://github.com/dongjuns/nano 
```
I use nanoAOD, and you need to set up the nanoAOD.

fork and clone it to your work station. such as gate, ui servers.

(1) nanoAOD setup

quick setup for analysis
```
scram p -n nanoAOD CMSSW CMSSW_9_4_4
cd nanoAOD/src
cmsenv
git clone git@github.com:CPLUOS/nano.git
scram b -j 20
getFiles
```

*if you have a problem at,
"git clone git@github.com:CPLUOS/nano.git"    

Do this.
```
#instead of
git clone git@github.com:CPLUOS/nano.git

#use        
git clone https://github.com/CPLUOS/nano.git
```

(2) dataset update

for update dataset,
```
#In the work station
cd /cms/scratch/yourusername/nanoAOD/slc6_amd64_gcc630/
./getDatasetInfo
```

brand new dataset .txt files will be made, 
```
/cms/scratch/yourusername/nanoAOD/src/nano/nanoAOD/data/dataset/
```


so far, this is environment setting.

all draw files are in the
```
cd /cms/scratch/jdj0715/nanoAOD/src/nano/analysis/test/topMass/
```

and there are 
```
topDraw.py  cmeson.py  plotMass.py  fitMass.py
```

I wll introduce them.

(1) topDraw.py & cmeson.py:

After ttbar event selecton and charmed meson selection,
I can make a control plots for Data/MC agreement check.

If this is your first time to use draw file,

```
#remove old DYFactor.json
rm DYFactor.json
#draw
python topDraw.py
#brand new DYFactor.json will be made, also.
```

but you have same samples,you don't need that process.

(2) plotMass.py:

There are various top quark mass MC samples, from 166.5 to 178.5
plotMass.py is used to make a binned histogram root file and json file.
so please make a invmass directory.
result of plotMass.py is will be located into invmass.

```
#make a invmass directory
mkdir invmass
#make binned histogram root file to the invmass directory
python plotMass.py
root -l invmass/ResultOfPlotmass.root
```

(3) fitMass.py:

Using "ResultOfPlotmass.json",
you can fit to the data and MC.
result of fit will be located into fres directory
so please make a fres directory
```
#make a fres directory
mkdir fres
#make a result of fit
python fitMass.py invmass/ResultOfPlotmass.json
display fres/fitResult.png
```

so far, these are draw files.
