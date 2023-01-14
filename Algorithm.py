import numpy as np
import pandas as pd

def preprocessing():

    # preprocessing of penguins
    Penguins = pd.read_csv("penguins.csv")
    # fill-in null values in gender with female
    # transform string values of gender into numeric
    Penguins["gender"].fillna(value="female", inplace=True)
    Penguins["gender"] = Penguins["gender"].replace(['female', 'male'], [1, 0])
    Penguins["species"] = Penguins["species"].replace(['Adelie', 'Gentoo','Chinstrap'], [0,1, 2])
    Train1 = Penguins.copy()
    for column in Train1.columns[1:]:
        Train1[column] = (Train1[column] - Train1[column].min()) / (
                Train1[column].max() - Train1[column].min())
    Train1.append(Penguins["species"])
    Penguins = np.array(Train1)

    return Penguins


def tanh(net):
    return (1 - np.exp(-1 * net)) / (1 + np.exp(-1 * net))

def Sigmoid(net):
    return 1/(1 + np.exp(-1*net))

def weights(Penguins,n, layers,function,bias,z,lr,inputsweight,hiddenweights,outputsweight,test):

    output=[]

    arr = Penguins[z, 1:6]

    if bias==1:
        arr= np.append(arr,1)

    for i in range(0,int(n[0])):
        if function =="Sigmoid":
            output.append(Sigmoid(np.dot(arr, inputsweight[i, :])))
        else:
            output.append(tanh(np.dot(arr, inputsweight[i, :])))


    netlist=[]
    for i in range(0,layers-1):
        netlist.append(output)
        if bias==1:
            output.append(1)
        out = []
        for x in range(0, int(n[i+1])):
            if function == "Sigmoid":
                out.append(Sigmoid(np.dot(hiddenweights[i][x, :], output)))
            else:
                out.append(tanh(np.dot(hiddenweights[i][x, :], output)))
        output = out

    netlist.append(output)
    finaloutput= []
    if bias==1:
        output.append(1)

    for i in range (0,3):
        if function == "Sigmoid":
            finaloutput.append(Sigmoid(np.dot(outputsweight[i, :], output)))
        else:
            finaloutput.append(tanh(np.dot(outputsweight[i, :], output)))

    netlist.append(finaloutput)
    max=finaloutput[0]
    index = 0
    outputs=[0,0,0]
    for i in range(1,3):

        if finaloutput[i]> max:
            max = finaloutput[i]
            index = i

    sigmas=[]
    outputs[index]=1

    hiddenweights.append(outputsweight)
    e=1
    if index != Penguins[z,0]:
        e=0
        if test == 1:
            return (inputsweight,hiddenweights,e,outputs)

        outputsigmas=[]
        for x in range(0, 3):
            if function == "Sigmoid":
                outputsigma = (Penguins[x][0] - netlist[len(netlist) - 1][x]) * netlist[len(netlist) - 1][x] * (
                            1 - netlist[len(netlist) - 1][x])
            else:
                outputsigma = (Penguins[x][0] - netlist[len(netlist) - 1][x]) * (1 + (netlist[len(netlist) - 1][x]))*(
                        1 - (netlist[len(netlist) - 1][x]))
            outputsigmas.append(outputsigma)
        sigmas.append(outputsigmas)

        for i in range((layers-1), -1, -1):
            hiddensigmas=[]
            for x in range(0, int(n[i])+bias):
                if function == "Sigmoid":
                    sigma = netlist[i][x] *(1 - netlist[i][x])
                else:
                    sigma = (1 + (netlist[i][x]))*(1 - (netlist[i][x]))

                s = 0
                for m in range (0,len(sigmas[len(sigmas)-1])):
                    s=s+(sigmas[len(sigmas)-1][m]*hiddenweights[i][m,x])

                sigma=sigma*s
                hiddensigmas.append(sigma)
            sigmas.append(hiddensigmas)

        sigmas.reverse()


        for i in range(0,int(n[0])):
            for z in range(0,5+bias):
                inputsweight[i,z] = inputsweight[i,z] +(arr[z] * sigmas[0][i]*lr)

        for i in range(0, layers-1):
            for x in range(0, int(n[i+1])+bias):
                for z in range(0, int(n[i])+bias):
                    hiddenweights[i][x,z]=hiddenweights[i][x,z]+(netlist[i][z]*sigmas[i+1][x]*lr)

        for x in range(0, 3):
            for z in range(0, int(n[len(n)-1]) + bias):
                hiddenweights[len(hiddenweights)-1][x, z] = hiddenweights[len(hiddenweights)-1][x, z] + (netlist[len(hiddenweights)-1][z] * sigmas[len(hiddenweights)][x] * lr)

    return (inputsweight,hiddenweights,e, outputs)

def confusion_matrix(actual, pred, num_of_samples):
    cm = np.zeros((3, 3))
    out =[]
    for i in range(num_of_samples):
        if actual[i][0] == 0:
            out.append([1,0,0])
        elif actual[i][0] == 1:
            out.append([0, 1, 0])
        elif actual[i][0] == 2:
            out.append([0, 0, 1])

    counter = 0
    # form an empty matric of 3x3
    for i in range(num_of_samples):
        # the confusion matrix is for 2 classes: 1,0
        # 1=positive, 0=negative
        if out[i] == pred[i]:
            counter+=1
            if actual[i][0] == 0:
                cm[0, 0] += 1
            elif actual[i][0] == 1:
                cm[1, 1] += 1
            elif actual[i][0] == 2:
                cm[2, 2] += 1
        elif out[i] != pred[i]:
            if actual[i][0] == 0 and pred[i][1] == 0:
                cm[0, 1] += 1
            elif actual[i][0] == 0 and pred[i][2] == 0:
                cm[0, 2] += 1
            elif actual[i][0] == 1 and pred[i][0] == 0:
                cm[1, 0] += 1
            elif actual[i][0] == 1 and pred[i][2] == 0:
                cm[1, 2] += 1
            elif actual[i][0] == 2 and pred[i][0] == 0:
                cm[2, 0] += 1
            elif actual[i][0] == 2 and pred[i][1] == 0:
                cm[2, 1] += 1

    print("Accuracy:", (counter / num_of_samples) * 100, "%")
    return cm


def training (n,l,bias,function,lr,e):
    epochs=int(e)

    penguins= preprocessing()
    Train=[]
    Train1= penguins[0:30]
    Train2=penguins[50:80]
    Train3=penguins[100:130]
    Train= np.vstack((Train1, Train2,Train3))
    np.random.shuffle(Train)


    layers=int(l)

    inputsweight = np.random.uniform(-1, 1, size=(int(n[0]), 5+bias))

    hiddenweights=[]
    for i in range(0,layers-1):
        hiddenweights.append(np.random.uniform(-1, 1, size=(int(n[i+1])+bias,int(n[i])+bias)))

    outputsweight = np.random.uniform(-1, 1, size=(3, int(n[len(n)-1])+bias))
    out =[]
    counter=0
    for x in range(0, epochs):
        for i in range(0,90):
            x,y,e,o =weights(Train, n, layers, function, bias, i, lr, inputsweight, hiddenweights, outputsweight,0)
            inputsweight=x
            hiddenweights=y[0:len(y)-1]
            outputsweight=y[len(y)-1]

    for i in range(0, 90):
        x, y, e, o = weights(Train, n, layers, function, bias, i, lr, inputsweight, hiddenweights, outputsweight, 1)
        out.append(o)
        counter = counter + e

    accuracy = (counter/(90)) *100
    print("Training Accuracy",accuracy,"%")
    c =confusion_matrix(Train, out, 90)
    print("Training Confusion Matrix:")
    print(c)



    return (inputsweight, hiddenweights,outputsweight)

def testing(n,layers,bias,function,lr,e):

    penguins = preprocessing()
    Test = []
    Test1 = penguins[30:50]
    Test2 = penguins[80:100]
    Test3 = penguins[130:150]
    Test = np.vstack((Test1, Test2, Test3))
    n=n.split(",")
    x,y,z=training(n,layers,bias,function,lr,e)
    counter=0
    out = []
    for i in range(0,60):
        m, d, e, o = weights(Test, n, layers, function, bias, i, lr, x, y, z, 1)
        out.append(o)
        counter=counter+e

    accuracy = (counter/60) *100
    print("Testing Accuracy", accuracy,"%")
    c =confusion_matrix(Test, out, 60)
    print("Testing Confusion Matrix:")
    print(c)

