#!/usr/bin/python
from dallantools import connect
from numpy import column_stack, array, vstack, zeros_like,savetxt, around
import sys
import os

conn = connect('selector')
c = conn.cursor()
trial = sys.argv[1]
stack = sys.argv[2]
threshold = sys.argv[3]

c.execute("""CREATE TEMPORARY TABLE real_probes (probe int unsigned, primary key (probe))""")
c.execute("""INSERT INTO real_probes (probe) SELECT probe FROM MSD WHERE trial=%s and stack=%s and t BETWEEN 0.03332 AND 0.03334 and r2 < %s""", (trial, stack, threshold))

c.execute("""SELECT distinct probe FROM real_probes""")
probes = [p[0] for p in c.fetchall()]
print len(probes), 'probes qualify'

c.execute("""select hour(start)*60+minute(start) from Stack where trial=%s and stack= %s""", (trial, stack))
minute, = c.fetchone()
label = 'T' + trial + '_' + str(minute) + 'min'
print label

c.execute("""SELECT DISTINCT t FROM MSD WHERE trial=%s AND stack=%s ORDER BY t""", (trial, stack))
times = array(c.fetchall()).T[0]
data = times
header = label + '_lagtime'
for probe in probes:
    c.execute("""SELECT r2 FROM MSD JOIN real_probes USING (probe) WHERE trial=%s AND stack=%s AND probe=%s""", (trial, stack, probe))
    this_data = array(c.fetchall()).T[0]
    padded_data = zeros_like(times)
    padded_data[:this_data.size] = this_data
    padded_data[this_data.size:] = 'NaN'
    data = vstack((data, padded_data))
    header += '\t' + label + '_P' + str(probe)

header += '\n'
print data.shape
filename = 'msd_' + label + '.txt'
savetxt(filename + '_tmp', data.T, fmt="%4.3e", delimiter="\t")

tempfile = open('temp.txt', 'w')
tempfile.write(header)
tempfile.close()
os.system("cat temp.txt " + filename + "_tmp > " + filename)
os.system("rm " + filename + "_tmp")
os.system("rm temp.txt")

