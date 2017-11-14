#!/usr/bin/python

import tkinter as tk
import os.path

configurationFile="config.txt"
root = tk.Tk()
root.title("Configuration For My Application")
root.geometry("500x80")
root.configure(background="white")

top = tk.Frame(root) # bg="light yellow" fill=tk.BOTH, expand=tk.TRUE
topl = tk.Frame(top)
topr = tk.Frame(top)
midL = tk.Frame(topr)
midR = tk.Frame(topr)
bottom = tk.Frame(root)

top.pack(side=tk.TOP)
bottom.pack(side=tk.BOTTOM)
topl.pack(side=tk.LEFT, padx=10 ,pady=10)
topr.pack(side=tk.RIGHT)
midL.pack(side=tk.LEFT)
midR.pack(side=tk.RIGHT)

lbl = tk.Label(root, text="Please select number of layers you want:")
lbl.configure(background="white")
lbl.pack(in_=topl, side=tk.LEFT)

def getValueFromFile(filename, keyval, defval):
    val = defval
    if not os.path.exists(filename):
        return defval

    in_file = open(filename, "rt")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        key, value = in_line.split(",")
        print(key+"="+value)
        if (key == keyval):
            val = value;
    in_file.close()
    return val

layers = tk.IntVar(root)
val = getValueFromFile(configurationFile,"layers",1)
layers.set(val) # initial value

def onPlus1Click(event=None):
    val = layers.get()
    if (val < 40):
        layers.set(val + 1)
def onMinus1Click(event=None):
    val = layers.get()
    if (val > 1):
        layers.set(val - 1)

plus1Button = tk.Button(root,text="+1", command=onPlus1Click, bg="white")
plus1Button.pack(in_=midL, side=tk.LEFT) # padx=5, pady=5)

option = tk.OptionMenu(root, layers, 1, 2, 3, 4, 5, 6, 7,8 ,9,10,
                                     11,12,13,14,15,16,17,18,19,20,
                                     21,22,23,24,25,26,27,28,29,30,
                                     31,32,33,34,35,36,37,38,39,40)
option.configure(background="orange")
option.pack(in_=midL, side=tk.RIGHT) # padx=5, pady=5)

minus1Button = tk.Button(root,text="-1", command=onMinus1Click, bg="white")
minus1Button.pack(in_=midR, side=tk.LEFT) # padx=5, pady=5)
#tk.Label(root, textvariable=counter).pack()
#tk.Label(root, textvariable=counter).pack()

def submit():
    layersFinalVal = layers.get()
    print ("value is "+str(layersFinalVal)+ "\n")
    with open(configurationFile, "wt") as outfile:
        outfile.write("layers,"+str(layersFinalVal)+"\n")
    outfile.close()
    root.quit()
button = tk.Button(root, text="Submit", command=submit)
button.configure(background="light blue")
button.pack(in_=bottom,side=tk.BOTTOM)

root.mainloop()
