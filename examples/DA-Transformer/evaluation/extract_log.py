import sys
import re
import argparse

res = {}
for line in sys.stdin.readlines():
    m = re.search(r"H-([0-9]+):?\s+(?:[\-0-9.infe]*)\s+(\S.*)$", line)
    if m:
        res[int(m.group(1))] = m.group(2).remove("## ", "")

for i in range(len(res)):
    print(res[i])
