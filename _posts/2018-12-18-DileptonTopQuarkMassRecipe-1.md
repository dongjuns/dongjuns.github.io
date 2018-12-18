---
title: "recipe : top quark mass in dilepton channel"
date: 2018-12-18 14:11:00 +0900
categories: TopQuarkMass
---
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

*if you have a problem at
"git clone git@github.com:CPLUOS/nano.git" ,

use
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

brand new dataset .txt files will be located, 
```
/cms/scratch/yourusername/nanoAOD/src/nano/nanoAOD/data/dataset/
```


so far, this is environment setting.
