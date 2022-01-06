import PIL
from PIL import Image
from PIL import ImageTk
from tkinter import *
from tkinter import ttk
from Source import *
from tkinter import filedialog


root = Tk()
root.title("Final Project: TGS Salt Identification")
root.config(bg="skyblue")
root.geometry("1210x700")
root.resizable(0, 0)

i = 0
tab_Panel = ttk.Notebook(root)
s = ttk.Style()
s.configure('tFrame.TFrame', background='pink')
bgcolor = "skyblue"
lpadx = 20
lpady = 5
bwidth = 15


def Resize(img, mask2=None, dsize=(220, 220)):
    if type(mask2) != type(None):
        mask2 = cv2.resize(mask2, (img.shape[0], img.shape[1]))
        for i in range(mask2.shape[0]):
            for j in range(mask2.shape[1]):
                if mask2[i][j] != 0:
                    img[i][j] = 150
    img = cv2.resize(img, dsize)
    img = PIL.Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return img

def Image_to_Dislay(img, x=220,y=220):
    dislay = cv2.resize(img, (x, y))
    dislay = cv2.cvtColor(dislay, cv2.COLOR_BGR2RGB)
    dislay = PIL.Image.fromarray(dislay)
    dislay = ImageTk.PhotoImage(dislay)
    return dislay
def Image_to_Background(img):
    dislay = cv2.resize(img, (1210, 700))
    dislay = cv2.cvtColor(dislay, cv2.COLOR_BGR2RGB)
    dislay = PIL.Image.fromarray(dislay)
    dislay = ImageTk.PhotoImage(dislay)
    return dislay
background_frame = Image_to_Background(cv2.imread("Background/background.jpg"))
holder_chroma_img = Image_to_Dislay(cv2.imread("Image_holder/Place_holder_img.jpg"),250,250)
holder_result = Image_to_Dislay(cv2.imread("Image_holder/Place_holder_result.jpg"),1150,250)

def choose_File_img():
    global img
    global img1
    global mask, mask2
    global label_img
    global cf1_Button_img
    global cl1_Button_img
    f = filedialog.askopenfile(initialdir="/", title="Select File",
                               filetypes=(("image", "*.jpg"), ("image", "*.png"),("image", "*.jpeg")))
    img = cv2.imread(f.name)
    img = cv2.resize(img, (101, 101))
    img1 = Image_to_Dislay(img,250,250)
    if f.name != f.name.replace("images", "masks"):
        try:
            mask = cv2.imread(f.name.replace("images", "masks"))
            mask2 = cv2.Canny(mask, 50, 150)
            mask = Image_to_Dislay(mask)
        except:
            mask2 = None
            mask = Image_to_Dislay(img, 220, 220)
    else:
        mask2 = None
        mask=Image_to_Dislay(img,220,220)


    label_img.grid_forget()
    label_img = Label(iFrame1, image=img1)
    label_img.grid(row=0, column=0, columnspan=2, padx=lpadx, pady=lpady)
    cf1_Button_img = Button(iFrame1, text="Choose file", command=lambda: choose_File_img(), width=bwidth)
    cf1_Button_img.grid(row=1, column=0)
    cl1_Button_img = Button(iFrame1, text="Clear", command=lambda: clear_img(), width=bwidth)
    cl1_Button_img.grid(row=1, column=1)

def clear_img():
    global label_img
    global cf1_Button_img
    global cl1_Button_img
    label_img.grid_forget()
    label_img = Label(iFrame1, image=holder_chroma_img)
    label_img.grid(row=0, column=0, columnspan=2, padx=lpadx, pady=lpady)
    cf1_Button_img = Button(iFrame1, text="Choose file", command=lambda: choose_File_img(), width=bwidth)
    cf1_Button_img.grid(row=1, column=0)
    cl1_Button_img = Button(iFrame1, text="Clear", command=lambda: clear_img(), width=bwidth)
    cl1_Button_img.grid(row=1, column=1)
def submit():
    global img, background
    global label_result, label_text, sm_Button, clr_Button
    global mask2
    global display_text1,display_text2,display_text3,display_text4,display_text5
    global res1, res2, res3, res4
    global res1_f, res2_f, res3_f, res4_f
    global mask

    res1 = Run(img, 112)
    res2 = Run(img, 5128)
    res3 = Run(img, 6128)
    res4 = Run(img, 9999)
    res1_f = Resize(res1, mask2)
    res2_f = Resize(res2, mask2)
    res3_f = Resize(res3, mask2)
    res4_f = Resize(res4, mask2)

    display_text1 = StringVar()
    display_text2 = StringVar()
    display_text3 = StringVar()
    display_text4 = StringVar()
    display_text5 = StringVar()
    display_text1.set("UNET 112")
    display_text2.set("UNET 5 101")
    display_text3.set("UNET 6 128")
    display_text4.set("ResUnet 101")

    if type(mask2) != type(None):
        display_text5.set("MASK")
    else:
        display_text5.set("Image resize")

    label_result.grid_forget()
    label_result = Label(iFrame2, image=res1_f)
    label_result.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
    label_result = Label(iFrame2, image=res2_f)
    label_result.grid(row=0, column=2, columnspan=2, padx=5, pady=5)
    label_result = Label(iFrame2, image=res3_f)
    label_result.grid(row=0, column=4, columnspan=2, padx=5, pady=5)
    label_result = Label(iFrame2, image=res4_f)
    label_result.grid(row=0, column=8, columnspan=2, padx=5, pady=5)
    label_result = Label(iFrame2, image=mask)
    label_result.grid(row=0, column=16, columnspan=2, padx=5, pady=5)

    label_text.grid_forget()
    label_text = Label(iFrame2, textvariable=display_text1)
    label_text.grid(row=1, column=0, columnspan=2, padx=lpadx, pady=lpady)
    label_text = Label(iFrame2, textvariable=display_text2)
    label_text.grid(row=1, column=2, columnspan=2, padx=lpadx, pady=lpady)
    label_text = Label(iFrame2, textvariable=display_text3)
    label_text.grid(row=1, column=4, columnspan=2, padx=lpadx, pady=lpady)
    label_text = Label(iFrame2, textvariable=display_text4)
    label_text.grid(row=1, column=8, columnspan=2, padx=lpadx, pady=lpady)
    label_text = Label(iFrame2, textvariable=display_text5)
    label_text.grid(row=1, column=16, columnspan=2, padx=lpadx, pady=lpady)

    sm_Button.grid_forget()
    sm_Button = Button(iFrame2, text="Submit", command=lambda: submit(), width=bwidth)
    sm_Button.grid(row=2, column=3)
    clr_Button.grid_forget()
    clr_Button = Button(iFrame2, text="Clear", command=lambda: clear_result(), width=bwidth)
    clr_Button.grid(row=2, column=8)

def clear_result():
    global label_result, label_text
    global sm_Button
    global clr_Button

    label_result.grid_forget()
    label_result = Label(iFrame2, image=holder_result)
    label_result.grid(row=0, column=0, columnspan=2, padx=20, pady=5)

    display_text1 = StringVar()
    display_text1.set("Waiting!")

    label_text.grid_forget()
    label_text = Label(iFrame2, textvariable=display_text1)
    label_text.grid(row=1, column=0, columnspan=2, padx=lpadx, pady=lpady)

    sm_Button.grid_forget()
    sm_Button = Button(iFrame2, text="Submit", command=lambda: submit(), width=bwidth)
    sm_Button.grid(row=1, column=0)
    clr_Button.grid_forget()
    clr_Button = Button(iFrame2, text="Clear", command=lambda: clear_result(), width=bwidth)
    clr_Button.grid(row=1, column=1)

#########################################

tab_img_img = ttk.Frame(tab_Panel, style="tFrame.TFrame")
background_label1 = Label(tab_img_img, image=background_frame)
background_label1.place(x=0, y=0, relwidth=1, relheight=1)
#Frame chromakey image : iFrame1
iFrame1 = Frame(tab_img_img, width=300, height=300, bg=bgcolor)
iFrame1.grid(row=0, column=0, padx=25, pady=15)
iFrame1.grid_propagate(0)

label_img = Label(iFrame1, image=holder_chroma_img)
label_img.grid(row=0, column=0, columnspan=2, padx=lpadx, pady=lpady)
cf1_Button_img = Button(iFrame1, text="Choose file", command=lambda: choose_File_img(), width=bwidth)
cf1_Button_img.grid(row=1, column=0)
cl1_Button_img = Button(iFrame1, text="Clear", command=lambda: clear_img(), width=bwidth)
cl1_Button_img.grid(row=1, column=1)


#Frame result : iFrame2
display_text1 = StringVar()
display_text1.set("Waiting!")


iFrame2 = Frame(tab_img_img, width=1190, height=300, bg=bgcolor)
iFrame2.grid(row=1, column=0, rowspan=2, padx=10, pady=10)
iFrame2.grid_propagate(0)

label_result = Label(iFrame2, image=holder_result)
label_result.grid(row=0, column=0, columnspan=2, padx=lpadx, pady=lpady)

label_text = Label(iFrame2, textvariable=display_text1)
label_text.grid(row=1, column=0, columnspan=2, padx=lpadx, pady=lpady)

sm_Button = Button(iFrame2, text="Submit", command=lambda: submit(), width=bwidth)
sm_Button.grid(row=1, column=0)
clr_Button = Button(iFrame2, text="Clear", command=lambda: clear_result(), width=bwidth)
clr_Button.grid(row=1, column=1)


tab_Panel.add(tab_img_img, text='Image and image')
tab_Panel.pack(expand=1, fill="both")

root.mainloop()