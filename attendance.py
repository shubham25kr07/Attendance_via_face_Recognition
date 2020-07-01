import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox as mb ,Text
import tkinter.font as font
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
import csv 
import xlsxwriter
import xlrd
import subprocess, sys

# import platform

# platform.platform()

window=tk.Tk()
window.title("Attendance_via_Face")
window.configure(background='light blue')
window.geometry("800x850")
window.resizable(0, 0) 

heading="ATTENDANCE VIA FACE RECOGNITION"

def isinteger(roll):
	try:
		int(roll)
		return True
	except:
		mb.showinfo("Message","Enter integer ID")
		return False


csvpath="studentdata/studentdata.csv"

def Save_Student_data(name,sec,roll):
	with open(csvpath, 'a+') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow([name,sec,roll])



def TakePhoto():

	#roll ==id

	name=(input1.get())
	sec=(input2.get())
	roll=(input3.get())

	if(name.isalpha() & sec.isalpha() & isinteger(roll)):

		Save_Student_data(name,sec,roll)
		
		faceCascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
		cap = cv2.VideoCapture(0)

		count = 0
		while(True):
			ret, img = cap.read()
			img = cv2.flip(img,1)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30))

			for(x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				count=count+1
				cv2.imwrite("dataset/"+str(name)+'.'+ str(roll)+'.'+str(sec) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
				cv2.imshow('CAMERA',img)

			if cv2.waitKey(30) & 0xFF== ord('q'):
				break
			elif count>=20:
				break

		cap.release()
		cv2.destroyAllWindows()
	else:
		if(not name.isalpha()):
			mb.showinfo("Message","enter valid Name")
		if(not sec.isalpha()):
			mb.showinfo("Message","enter valid  Sec")

def TrainData():
	# Path for face image database
	mb.showinfo("Message","Might take few sec")

	path = 'dataset'
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	detector = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

	faces,ids = getImagesAndLabels(path,detector)
	recognizer.train(faces, np.array(ids))
	recognizer.write('trainer/trainer.yml') 

	mb.showinfo("Message","Data saved successfully")

	input1.delete(0,'end')
	input2.delete(0,'end')
	input3.delete(0,'end')
	
# function to get the images and label data
def getImagesAndLabels(path,detector):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids


def save_student_present_record(id,name,sec,path):
	with open(path, 'a+') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow([id,name,sec,"P"])

def DetectImage():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read('trainer/trainer.yml')

	faceCascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	cam = cv2.VideoCapture(0)
	cam.set(3, 640) # set video widht
	cam.set(4, 480) # set video height


	# Define min window size to be recognized as a face
	minW = 0.1*cam.get(3)	
	minH = 0.1*cam.get(4)
	id= -1
	names=""
	sect=""

	while True:
		ret, img =cam.read()
		img = cv2.flip(img, 1) # Flip vertically
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(int(minW),int(minH)))

		
		for(x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
			id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
			# If confidence is less them 100 ==> "0" : perfect match 
			if (confidence < 100):
				confidence = "  {0}%".format(round(100 - confidence))
			else:
				confidence = "  {0}%".format(round(100 - confidence))
				id=-1

			data=pd.read_csv(csvpath,index_col="Id")
			rows=data.loc[[id]]
			colnames=['Name','Sec','Id']
			name=rows[colnames[0]].values
			sec=rows[colnames[1]].values
			names=name
			sect=sec
			#print(name[0])
			#print(sec)
			#print(rows[colnames[1]])

			cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
			cv2.putText(img,name[0],(x+50,y-5),font,1,(255,255,255),2)
			cv2.putText(img,sec[0],(x+5,y+h-5),font,1,(255,255,255),2)
		
		
		cv2.imshow('camera',img)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	loc='studentdata/recorddata.csv'

	save_student_present_record(id,names[0],sect[0],loc)

	cam.release()
	cv2.destroyAllWindows()


def OpenAttendance():
	opener ="open" if sys.platform == "darwin" else "xdg-open"
	path=os.getcwd()+"/studentdata/recorddata.csv"
	subprocess.call([opener,path])

label=tk.Label(window,text=heading,fg="grey",width=67,height=4, font=("Helvetica", 16),justify="center")
label.place(x=0,y=10)

label0l=tk.Label(window,text="NOT REGISTERED USERS",fg="grey",width=30,height=4, font=("Helvetica", 16),justify="center")
label0l.place(x=0,y=150)

label0r=tk.Label(window,text="REGISTERED USERS",fg="grey",width=30,height=4, font=("Helvetica", 16),justify="center")
label0r.place(x=440,y=150)

label1=tk.Label(window,text="Name :",bg="light blue",fg="blue",width=10,height=2, font=("Helvetica", 16,'bold'),justify="left")
label1.place(x=0,y=300)
input1 = tk.Entry(window,width=20  ,bg="light green" ,fg="black",font=('times', 15, ' bold '))
input1.place(x=120, y=310)

label2=tk.Label(window,text="Sec:",bg="light blue",fg="blue",width=10,height=2, font=("Helvetica", 16,'bold'),justify="left")
label2.place(x=0,y=380)
input2 = tk.Entry(window,width=20  ,bg="light green" ,fg="black",font=('times', 15, ' bold '))
input2.place(x=120, y=390)

label3=tk.Label(window,text="ID:",bg="light blue",fg="blue",width=10,height=2, font=("Helvetica", 16,'bold'),justify="left")
label3.place(x=0,y=455)
input3 = tk.Entry(window,width=20  ,bg="light green" ,fg="black",font=('times', 15, ' bold '))
input3.place(x=120, y=465)

label1l=tk.Label(window,text="Face Recognition",fg="grey",width=20,height=2, font=("Helvetica", 16),justify="left")
label1l.place(x=0,y=525)

button1=tk.Button(window,text="Photo",command=TakePhoto,bg="light blue",fg="blue",width=6,height=1, font=("Helvetica", 16,'bold'),justify="center")
button1.place(x=30,y=600)

label2l=tk.Label(window,text="Save Data",fg="grey",width=20,height=2, font=("Helvetica", 16),justify="left")
label2l.place(x=0,y=665)

button2=tk.Button(window,text="Save",command=TrainData,bg="light blue",fg="blue",width=6,height=1, font=("Helvetica", 16,'bold'),justify="center")
button2.place(x=30,y=740)

label1r=tk.Label(window,text="Mark Attendance",fg="grey",width=20,height=2, font=("Helvetica", 16),justify="left")
label1r.place(x=555,y=300)

button3=tk.Button(window,text="Attendance",command=DetectImage,bg="light blue",fg="blue",width=8,height=1, font=("Helvetica", 16,'bold'),justify="center")
button3.place(x=600,y=380)

label2r=tk.Label(window,text="Check Attendance",fg="grey",width=20,height=2, font=("Helvetica", 16),justify="left")
label2r.place(x=555,y=450)

button3=tk.Button(window,text="Open",command=OpenAttendance,bg="light blue",fg="blue",width=8,height=1, font=("Helvetica", 16,'bold'),justify="center")
button3.place(x=600,y=530)

window.mainloop()
