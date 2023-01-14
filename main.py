from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import Algorithm

def interface():
    master = Tk()
    master.title("Task 3")
    image0 = Image.open('neural-network-background.png')
    background = ImageTk.PhotoImage(image0)
    master.geometry('1000x600')
    canvas = Canvas(master)
    canvas.create_image(0, 0, image=background, anchor=NW)
    canvas.create_text(50, 70, text="Number of hidden layer", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)
    canvas.create_text(50, 160, text="Number of neurons", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)
    canvas.create_text(50, 250, text="Learning Rate", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)
    canvas.create_text(50, 340, text="Epochs", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)
    canvas.create_text(50, 430, text="Activation Functon", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)
    canvas.create_text(50, 520, text="Add Bias", fill="white", font=("Helvetica", 14, 'bold'), anchor=NW)


    HiddenLayers = Entry(master)
    HiddenLayers.place(x=300, y=70, width=150)
    Neurons = Entry(master)
    Neurons.place(x=300, y=160, width=150)
    LearningRate = Entry(master, font=("Helvetica", 10))
    LearningRate.place(x=300, y=250, width=150)
    Epochs = Entry(master, font=("Helvetica", 10))
    Epochs.place(x=300, y=340, width=150)
    Function = ["Sigmoid", "Hyperbolic Tangent sigmoid "]
    function = ttk.Combobox(master, value=Function)
    function.place(x=300, y=430, width=150)
    bias = IntVar()
    add_bias = Checkbutton(master, text="Add", variable=bias, bg="white", fg="black", width=10,
                           font=("Helvetica", 12, 'bold'))
    add_bias.place(x=300, y=520, width=150)

    penguins = Algorithm.preprocessing()

    b = Button(master, text="Enter",  bg="white", fg="black",font=("Helvetica", 12, 'bold'), command=lambda: Algorithm.testing(Neurons.get(),int(HiddenLayers.get()),int(bias.get()),function.get(),float(LearningRate.get()),float(Epochs.get())))
    b.place(x=800, y=520, width=130)

    canvas.pack(fill="both", expand=True)


    master.mainloop()

interface()






