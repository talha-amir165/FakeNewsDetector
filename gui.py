
from tkinter import ttk
from tkinter import *
import tkinter as tk
from turtle import heading
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
window = Tk()
backgroundcolor = '#2c2d30'
#frame = tk.Frame(master=window, width=150, height=150)
#frame.pack()

window.columnconfigure(0, minsize=250)
window.rowconfigure([0, 1], minsize=100)


enter=tk.StringVar(window)
enter.set("Hello!here our text will be printed")
#def predict():
 #   global enter
 #   txt.destroy()
 #   txt2=Label(master=window,text=enter,height=3,width=10,bg='white')
 #   txt2.grid(row=4)

def predict(l1,l2):
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    # tfidf_test= tfidf_vectorizer2.fit_transform([l2])
      
    pac52 = joblib.load(r'D:\ML\Fake_News_Detection-main\regression.joblib')
    # result=pac52.predict(tfidf_test2)
    


    if __name__=="__main__":
        
    # Text to classify should be in a list.
        text = ["The move would make it difficult for the Trump administration to demolish the exchanges.",
                "It was supposed to end like this for Marco Rubio.", 
                "Prime Minister Benjamin Netanyahu on Sunday said he would not allow Israel to be submerged by refugees after calls for the Jewish state to take in those fleeing Syria's war.",
                "Mr. Kehinde, what are you doing next? this is great!"]
        textdata = tfidf_vectorizer.fit_transform(text)
        news = pac52.predict(textdata)
        data = []
        for text, pred in zip(text, news):
            data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(data, columns = ['text','Prediction'])
        df = df.replace([0,1], ["FAKE","REAL"]) #Replacing the class of 0 and 1 with Negative and Positive respectively
        print(df.head())
    
    
    return result


h = Label(master=window,font=("Cambria", 25, "bold"), text='Machine Learning Project')
l1= Label(master=window,font=("Cambria", 15, "bold"), text='Enter news title')
l2= Label(master=window, font=("Cambria", 15, "bold"),text='Enter news text')
h.grid(row=0,padx=0,column=1)
l1.grid(row=1)
l2.grid(row=2)

e1 = ttk.Entry(master=window, font=("Cambria", 15, "bold"), width = 20)
e2 = ttk.Entry(master=window, font=("Cambria", 15, "bold"), width = 20)
e1.grid(row=1, column=1,padx=30, pady=5)
e2.grid(row=2, column=1,padx=30, pady=5)

b = Button(master=window, text='Predict', height=1,width=6, font=("Cambria", 18), foreground="white",activeforeground="black", borderwidth=6, background="#000000",activebackground="#03fc24",command=lambda:enter.set(predict(e1.get(),e2.get())))
b.grid(row=3, column=3, padx=0, pady=5)

txt=Label(master=window,textvariable=enter,height=10,width=20,bg='white')
txt.grid(row=4,columnspan=7,sticky='ew')


window.title("ML Project")

window.configure(bg=backgroundcolor)
window.mainloop()