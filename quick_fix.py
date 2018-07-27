def get(input_file, output_file):
    f = open(input_file, "r")
    o = open(output_file, "w")
    for line in f:
        list = line.split('\t')
        o.write(
            list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
            list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[14] + '\t' + list[15] + '\t' + list[
                16] + '\t' + list[17] + '\t' + list[18] + '\t' + list[19] + '\t' + list[20] + '\n')
    f.close()
    o.close()

get('CohMetrixOutput.txt','CohMetrixOutput.txt')