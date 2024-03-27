from tkinter import *
import torch.cuda
from PIL import ImageTk,Image
from tkinter import filedialog
from predict import *

root = Tk()
root.title('BIKE VS CAR CLASSIFICATION')

def open():
        """

        :return:
        """
        global my_image
        root.filename = filedialog.askopenfilename(initialdir="C:/Users/parisa/Desktop/car_Vs_bike_classification/data/test/", title="Select A File", filetypes=(("jpeg files", "*.jpeg"),("all files", "*.*")))
        my_image = ImageTk.PhotoImage(Image.open(root.filename))
        my_image_label = Label(image=my_image)
        my_image_label.pack()
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CarBikeClassifier(num_classes=2)
        model.load_state_dict(torch.load('../pretrained_models/model.pth', map_location=torch.device('cpu')))
        #print(image_file)
        #prediction_text = predict_image(model, image_file, device='cpu')
        classify_result=predict_image(model, root.filename, device)
        if classify_result == 'bike':
            classify = Label(root, text=classify_result, font=("Helvetica", 20), bg="yellow",borderwidth=2, relief="groove")
            classify.pack(pady=10)
        else:
            classify = Label(root, text=classify_result,font=("Helvetica", 20), bg="sky blue",borderwidth=2, relief="groove")
            classify.pack(pady=10)

my_btn = Button(root, text="Select a picture", command=open).pack()
root.mainloop()
