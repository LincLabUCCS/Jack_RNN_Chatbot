def get(input_file, output_file):
    f = open(input_file, "r")
    o = open(output_file, "w")
    for line in f:
        if line[0] == "6" and line[1] not in ['0','1','2','3','4','5','6','7','8','9']:
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "13":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "15":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "19":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "44":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "47":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "50":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "68":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "73":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "92":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:2] == "96":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18] + '\n')
        if line[0:3] == "104":
            list = line.split('\t')
            o.write(
                list[3] + '\t' + list[4] + '\t' + list[5] + '\t' + list[6] + '\t' + list[7] + '\t' + list[8] + '\t' +
                list[9] + '\t' + list[10] + '\t' + list[11] + '\t' + list[12] + '\t' + list[13] + '\t' + list[
                    14] + '\t' + list[15] + '\t' + list[16] + '\t' + list[17] + '\t' + list[18])
    f.close()
    o.close()

get('CohMetrixOutput.txt','norm-2chat-sco.txt')