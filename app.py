import io
import random
from array import *

import joblib
import numpy as np
from flask import Flask, request, render_template, Response, make_response
from openpyxl import load_workbook
import pickle

import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import colorsys

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use('seaborn-talk')
import warnings
warnings.filterwarnings("ignore")

#Create an app object using the Flask class.
app = Flask(__name__)

### GLOBAL VARIABLES
globalacumulado = pd.DataFrame()
globalsemana = pd.DataFrame()

#Load the trained model. (Pickle file)
model = pickle.load(open('models/modelo_treinado_v1.sav', 'rb'))

#######

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/questions2')
def questions2():
    return render_template('questions2.html')

@app.route('/results3')
def results3():
    book = load_workbook("./training_files/TAT-SCRIPTS-HISTORICO-v3-4-QUINZENAL.xlsx")

    sheet = book.active

    return render_template('results3.html', sheet=sheet)

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # f
    result = prediction[0]

    resultfinal = "Semana: " + str(features[0][0]) + " , Scripts: " + str(round(result[0]))

    #########
    plot_png()

    return render_template('results.html', result=resultfinal)

@app.route('/static/images/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/", methods=["GET", "POST"])
def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    df = pd.read_excel("./training_files/TAT-SCRIPTS-HISTORICO-v3-4-QUINZENAL.xlsx", index_col=0)

    acumulado = df['acumulado']
    semana = df['semana']

    #xs = range(5)
    #ys = [random.randint(1, 5) for x in xs]
    #axis.plot(xs, ys)

    axis.plot(semana, acumulado)

    axis.set_title('PREVISÃO PARA SCRIPTS AUTOMATIZADOS QUINZENALMENTE')
    axis.set_xlabel('SEMANAS')
    axis.set_ylabel('SCRIPTS ACUMULADOS')

    #axis.title("First Weeks Accumulated Scripts")


    return fig

@app.route('/static/images/plot2.png/')
def plot_png2():

    fig = create_figure2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/", methods=["GET", "POST"])
def create_figure2():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    global globalacumulado
    global globalsemana

    #acumulado = result
    #semana = weeks

    axis.plot(globalsemana, globalacumulado)

    # axis.title("First Weeks Accumulated Scripts")

    axis.set_title('PREVISÃO PARA SCRIPTS AUTOMATIZADOS QUINZENALMENTE')
    axis.set_xlabel('SEMANAS')
    axis.set_ylabel('SCRIPTS ACUMULADOS')

    return fig

@app.route('/predict2',methods=['POST'])
def predict2():

    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model

    features2 = features
    #features2 = features2[2:]

    #print(features)
    #print(features2)

    a = []
    b = []
    c = []

    #c.append("")
    #b.append("")
    #a.append("")

    aux0 = features[0][0]
    aux1 = features[0][1]
    aux2 = features[0][2]
    aux3 = features[0][3]
    aux4 = features[0][4]
    aux5 = features[0][5]

    resultfinal1 = []
    resultfinal2 = []

    for k in range( aux1, aux2 ):

        # predicao_scripts= reg.predict(np.array([[k, 2, 8,16]]))
        # Colocado a média de scripts quinzenal para 2 pessoas que é 20.

        #a = model.predict( [ [ k, int(features[2]), int(features[3]), int(features[4]) ] )

        a = model.predict(np.array([[k, aux3, aux4, aux5]]))

        c.append(int(k))
        b.append(int(a[0][0]))

        resultfinal1 = " ( Semana: " + str(k) + " / Scripts: " + str(round(a[0][0])) + " )"

        resultfinal2.append(resultfinal1)

        if int(a[0]) >= int(aux0):
            break

    #prediction = model.predict(features)  # f
    #result = b[0]
    #result = features

    df = pd.DataFrame(b, columns=['acumulado'])
    df2 = pd.DataFrame(c, columns=['semana'])

    acumulado = df['acumulado']
    semana = df2['semana']

    global globalacumulado
    global globalsemana

    globalacumulado = acumulado
    globalsemana = semana

    result = b
    #weeks = c

    #########
    plot_png2()



    return render_template('results2.html', result=resultfinal2)

if __name__ == "__main__":
    app.run(debug=True)
