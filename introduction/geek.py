import re

rf = open('geek.txt', 'r')
lines = rf.readlines()
rf.close()

wf = open('geekOut.txt', 'w')
allCount = 0
text = []

pattern = re.compile('^\D+$', re.IGNORECASE | re.UNICODE)

for line in lines:
    x = line.split(' ')
    allCount += len(x)
    for y in x:
        if pattern.match(y):
            text.append(y)

unique = set(text)
for x in unique:
    print x
print 'Token Count ', allCount 
print 'Unique Count ', len(unique) 
