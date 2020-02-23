from flask import Flask
from flask import render_template
import pygal
import pandas as pd
from flask_ngrok import run_with_ngrok
from pygal.style import TurquoiseStyle
from pygal.style import DarkStyle

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route('/')
def pygalexample():

    df=pd.read_excel('data.xlsx') 
    conf=sum(list(df['Confirmed']))
    death=sum(list(df['Deaths']))
    recov=sum(list(df['Recovered']))
    
    df1 = pd.read_csv('death.csv')
    df2 = pd.read_csv('confirm.csv')
    df3 = pd.read_csv('recover.csv')
    df4=pd.read_csv('country.csv')

    graph = pygal.Line(style=DarkStyle)
    graph.title="Confirmed Cases"
    graph.x_labels=list(df2['Date'])
    graph.add('Confirmed Count',list(df2['Confirmed Count']))
    graph_data1=graph.render_data_uri()

    graph2 = pygal.Line(style=DarkStyle)
    graph2.title="Death Toll"
    graph2.x_labels=list(df2['Date'])
    graph2.add('Death Count',list(df1['Death Count']))
    graph_data2=graph2.render_data_uri()

    graph3 = pygal.Line(style=DarkStyle)
    graph3.title="Recovered Cases"
    graph3.x_labels=list(df2['Date'])
    graph3.add('Death Count',list(df3['Recovery']))
    graph_data3=graph3.render_data_uri()

    line_chart = pygal.Line(style=DarkStyle)
    line_chart.title = 'Aggregate'
    line_chart.x_labels = list(df2['Date'])
    line_chart.add('Confirmed', list(df2['Confirmed Count']))
    line_chart.add('Death', list(df1['Death Count']))
    line_chart.add('Recovered',list(df3['Recovery']))
    chart4=line_chart.render_data_uri()

    worldmap_chart = pygal.maps.world.World(style=DarkStyle)
    worldmap_chart.title = 'Density Distribution'
    worldmap_chart.add('Corona Density', {
    'au': 656,
    'be': 34,
    'ca': 280,
    'eg': 14,
    'fi': 46,
    'fr': 442,
    'de': 596,
    'hk': 1722,
    'in':3,
    'ir': 14,
    'it': 112,
    'jp': 1671,
    'mo': 475,
    'cn': 31940,
    'my': 728,
    'np': 54,
    'ph': 110,
    'ru': 84,
    'sg':2055,
    'es':64,
    'lk':50,
    'se':42,
    'tw':778,
    'th':1367,
    'gb':240,
    'ae':306,
    'us':586,
    'vn':542

    })
    map1=worldmap_chart.render_data_uri()
    
    return render_template("hello.html",chart=graph_data1,chart2=graph_data2,chart3=graph_data3,chart4=chart4,map1=map1,val1=conf,val2=death,val3=recov)

