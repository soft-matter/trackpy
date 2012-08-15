#!/usr/bin/python
from dallantools import connect
from numpy import column_stack, array, vstack, zeros_like,savetxt, around
import sys
import os

conn = connect('selector')
c = conn.cursor()
trial = sys.argv[1]
stack = sys.argv[2]

c.execute("""SELECT distinct probe FROM MSD where trial=%s and stack=%s""", (trial, stack))
probe_count = c.rowcount

c.execute("""select hour(start)*60+minute(start) from Stack where trial=%s and stack= %s""", (trial, stack))
minute, = c.fetchone()
label = 'T' + trial + '_' + str(minute) + 'min'
print label

c.execute("""SELECT DISTINCT t FROM MSD where trial=%s and stack=%s""", (trial, stack))
times = array(c.fetchall()).T[0]
data = times
header = label + '_lagtime'
for probe in range(probe_count):
    c.execute("""SELECT r2 FROM MSD where trial=%s and stack=%s and probe=%s""", (trial, stack, probe))
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

