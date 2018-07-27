import random


def truncator(text, output):

    f = open(text, 'r')
    o = open(output, 'w')

    num_lines = 0

    for line in f:
        if num_lines == 10000:
            f.close()
            o.close()
            break
        else:
            num_lines += 1
            o.write(line)

def random2(text):
    f = open(text, 'r')
    start = random.randint(1, 9970)
    finish = start + 30
    counter = 1
    for line in f:
        if (start <= counter <= finish) == True:
            counter += 1
            print (line)

random2('data/scotus_10000/scotus_trunc_10000.txt')