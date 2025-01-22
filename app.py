from flask import Flask, render_template, request,url_for
import pickle
import numpy as np

model = pickle.load(open('svm.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def info():
    return render_template('homepage.html')
    
@app.route('/calci')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['field1']
    data=list(data1.split(','))
    # data2 = int(request.form['field2'])
    # data3 = int(request.form['field3'])
    # data4 = int(request.form['field4'])
    # data5 = int(request.form['field5'])
    # data6 = int(request.form['field6'])
    # data7 = int(request.form['field7'])
    # data8 = int(request.form['field8'])
    # data9 = int(request.form['field9'])
    # data10 = int(request.form['field10'])
    # data11= int(request.form['field11'])
    # data12= int(request.form['field12'])
    # data13= int(request.form['field13'])
    # data14= int(request.form['field14'])
    # data15= int(request.form['field15'])
    # data16= int(request.form['field16'])
    # data17= int(request.form['field17'])
    # data18= int(request.form['field18'])
    # data19= int(request.form['field19'])
    # data20= int(request.form['field20'])
    # data21= int(request.form['field21'])
    # data22= int(request.form['field22'])
    # data23= int(request.form['field23'])
    # data24= int(request.form['field24'])
    # data25= int(request.form['field25'])
    arr = np.array([data])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

@app.route('/schemes')
def scheme():
    return render_template('scheme.html')

if __name__ == "__main__":
    app.run(debug=True)
