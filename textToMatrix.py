def textToMatrix():
    text = input("Enter the matrix in text format: ")
    return [[float(j) for j in i.split()] for i in text.split(';')]


while(1):
    try:
        print("input matrix =\n", textToMatrix())
    except:
        print("Error in input matrix")