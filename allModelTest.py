from argparse import ArgumentParser

ap=ArgumentParser()
ap.add_argument('-i',help='Input glob expression for meta files to be tested... default is all in current directory (be sure to use quotes for e.g. "*.meta")')
parsed=ap.parse_args()
if parsed.i==None:
    inGlob="*.meta"
else:
    inGlob=parsed.i

from glob import glob
from itertools import chain
import re,os,sys

models=[re.compile('(.+)\.meta').findall(tmp)[0] for tmp in glob(inGlob)]

print("Testing the following models:")
print('\n'.join(models))
sys.stdout.flush()

cmds_a=['python test_accuracy.py -v -m '+tmp+' -i testSets/PsychologyEth.pkl 2> /dev/null' for tmp in models]
cmds_b=['python test_accuracy.py -v -m '+tmp+' 2> /dev/null' for tmp in models]
cmds=list(chain(*zip(cmds_a,cmds_b)))

for cmd in cmds:
    os.system(cmd)
    sys.stdout.flush()
