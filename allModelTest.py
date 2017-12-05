from glob import glob
from itertools import chain
import re,os,sys

models=[re.compile('(.+)\.meta').findall(tmp)[0] for tmp in glob('currentBest_*.meta')]
cmds_a=['python test_accuracy.py -v -m '+tmp+' -i testSets/PsychologyEth.pkl 2> /dev/null' for tmp in models]
cmds_b=['python test_accuracy.py -v -m '+tmp+' 2> /dev/null' for tmp in models]
cmds=list(chain(*zip(cmds_a,cmds_b)))

for cmd in cmds:
    os.system(cmd)
    sys.stdout.flush()
