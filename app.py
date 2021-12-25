import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
MainNotebook_pkl2=pickle.load(open('MainNotebook_pkl2','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    
    feature_list=[]

    for i in int_features:
        if(i=="Science"):
            feature_list.append(0)
            feature_list.append(0)
            feature_list.append(1)

        elif(i=="Commerce"):
            feature_list.append(0)
            feature_list.append(1)
            feature_list.append(0)
        elif(i=="Arts"):
            feature_list.append(1)
            feature_list.append(0)
            feature_list.append(0)
        elif(i=="Sci&Tech"):
            feature_list.append(0)
            feature_list.append(0)
            feature_list.append(1)
        elif(i=="Comm&Mgmt"):
            feature_list.append(1)
            feature_list.append(0)
            feature_list.append(0)
        elif(i=="Others"):
            feature_list.append(0)
            feature_list.append(1)
            feature_list.append(0)  
        else:
            
            feature_list.append(i)
    


    final_features = [np.array(feature_list)]
    
    

    prediction = MainNotebook_pkl2.predict(final_features)
    
    return render_template('result.html',prediction=prediction)




app.run(debug=True)