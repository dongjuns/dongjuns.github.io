---
title: "recipe : top quark mass in dilepton channel (2)"
date: 2018-12-18 14:40:00 +0900
categories: TopQuarkMass
---
all dra files are in the
```
cd /cms/scratch/jdj0715/nanoAOD/src/nano/analysis/test/topMass/
```

and there are 
```
topDraw.py  cmeson.py  plotMass.py  fitMass.py
```

I wll introduce them.

(1) topDraw.py & cmeson.py
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

but you have same samples,
you don't need that process.

(2) plotMass.py
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

(3) fitMass.py
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
