text = 'shakespeare.txt'

output = 'output_file.txt'

f = open(output, 'r')
o = open(text, 'w')



for line in f:
    o.write(line)

f.close()
o.close()