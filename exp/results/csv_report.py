import pandas as pd
import numpy as np
import sys


def classification_report_df(report, id):
    lines = report.split('\n')
    r = []
    r.append([''] + lines[0].split() + ['id'])
    r.append([lines[2].split()[0]] + [float(x)
             for x in lines[2].split()[1:]] + [id])
    r.append([lines[3].split()[0]] + [float(x)
             for x in lines[3].split()[1:]] + [id])
    r.append([lines[5].split()[0]] + [0, 0] + [float(x)
             for x in lines[5].split()[1:]] + [id])
    r.append([lines[6].split()[0]] + [float(x)
             for x in lines[6].split()[2:]] + [id])
    r.append([lines[7].split()[0]] + [float(x)
             for x in lines[7].split()[2:]] + [id])
    r = np.array(r)
    return pd.DataFrame(data=r[1:, 1:], index=r[1:, 0], columns=r[0, 1:])


arq = sys.argv[1]
arq_out = sys.argv[2]
#arq = '/home/thiagodepaulo/exp/PyPBG/exp/results/CSTR/out_1_1'
r = False
other = []
robotics = []
report = ''
l_df = []
id = 0
for line in open(arq, 'r'):
    s = line.split()
    if len(s) >= 1 and s[0] == 'topic':
        if r:
            l_df.append(classification_report_df(report, id))
            id += 1
        r = False
        report = ''
    if len(s) >= 1 and s[0] == 'precision':
        r = True
    if r:
        report += line

df = pd.concat(l_df)
df.to_csv(arq_out)
