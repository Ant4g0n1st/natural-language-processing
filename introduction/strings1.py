#Open file
rf = open('input.txt', 'r')
#
wf = open('output.txt', 'w')
lines = rf.readlines()
rf.close()

vowels = ['a', 'e', 'i', 'o', 'u']
text = []
for x in lines:
    z = x.split(' ')
    newLine = []
    for y in z:
        y = y.strip()[1 : ]
        y = ''.join(['#', y])
        if y[-1] in vowels:
            y = ''.join([y, '$VOC'])
        else:
            y = ''.join([y, '$CONS'])
        newLine.append(y)
    wf.write(' '.join(newLine))
    text.append(' '.join(newLine)[::-1])
wf.write('\n')
for x in text:
    wf.write(x) 
wf.close()

