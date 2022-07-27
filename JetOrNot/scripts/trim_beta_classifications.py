import numpy as np
import datetime
from dateutil.parser import parse
import sys

fname = sys.argv[1]

assert not fname is None, "Please provide a filename"

with open(fname, 'r') as infile:
    lines = infile.readlines()

launch_date = parse('2021-12-07 00:00:00 UTC')

with open('question_extractor_trimmed.csv', 'w') as outfile:
    # write out the header
    outfile.write(lines[0])
    for line in lines[1:]:
        datei = parse(line.split(',')[5])
        if datei >= launch_date:
            outfile.write(line)
