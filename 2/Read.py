if __name__ == "__main__":
    inputfile = open("submission.csv", "rb")
    for line in inputfile:
        print(line)
    inputfile.close()