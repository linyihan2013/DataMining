lines = open('train/train.csv').readlines()
output = open('train/sort.csv', 'w')
result = []

for line in lines:
    line = line.strip()
    result.append([line.split(',')[0], line.split(',')[1]])

result = sorted(result, key=lambda x:(len(x[0]), x[0], x[1]))
for line in result:
    output.write(("%s,%s\n") % (line[0], line[1]))
output.close()