if __name__ == "__main__":
    #   Preprocess for train
    partnums = 200
    inputfile = open("train.txt", "rb")
    outputlist = []
    for i in range(partnums):
        outputlist.append(open('train/part%04d'%i,'wb'))

    index = 0
    while True:
        chunk = inputfile.readline()
        if not chunk:
            break
        outputlist[index].write(chunk)
        index += 1
        if index == partnums:
            index = 0

    for i in range(partnums):
        outputlist[i].close()

    #   Preprocess for test
    # partnums = 20
    # inputfile = open("test.txt", "rb")
    # outputlist = []
    # for i in range(partnums):
    #     outputlist.append(open('test/part%04d'%i,'wb'))
    #
    # index = 0
    # while True:
    #     chunk = inputfile.readline()
    #     if not chunk:
    #         break
    #     outputlist[index].write(chunk)
    #     index += 1
    #     if index == partnums:
    #         index = 0
    #
    # for i in range(partnums):
    #     outputlist[i].close()