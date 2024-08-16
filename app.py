from concurrent.futures import thread
from flask import Flask, flash, redirect, render_template,request,url_for
from werkzeug.utils import secure_filename
import os
from colorthief import ColorThief
import pandas as pd
from scipy.spatial import KDTree
from webcolors import hex_to_rgb,CSS3_HEX_TO_NAMES
import zipfile
from PIL import Image
from math import sqrt

from clothes_detection.new_image_demo import *
from rb import *
from mask.demo import *


app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_FOLDER = 'static/data/'
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_file():
    dir = 'static/data/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
def remove_class():
    dir='static/detection/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))
def remove_seg():
    dir='static/segmentation/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))
def remove_mask():
    dir='static/cnn_out/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))

def directory():
    path = "static/data/"
    dir = os.listdir(path)
    return dir

def detect_dir():
    path2="static/detection/"
    dir2=os.listdir(path2)
    return dir2
def seg_dir():
    path="static/segmentation/"
    dir=os.listdir(path)
    return dir
def mask_dir():
    path="static/cnn_out/"
    dir=os.listdir(path)
    return dir

def define_main_color(variable):
    rgb=[]
    css2=[]
    css3=[]
    css_2_arr_pal=[]
    css_3_arr_pal=[]
    count_img=0
    for i in variable:
        count=0
        link="static/segmentation/"+i
        print(i)
        im = Image.open(link)
        pix_val = list(im.getdata())
        for i in pix_val:
            if i!=(0,0,0,0):
                count+=1
        if count==0:
            roster=directory()
            link2="static/data/"+roster[count_img]
            print(link2)
            color_thief=ColorThief(link2)
            dominant_color=color_thief.get_color(quality=1)
            #palette = color_thief.get_palette(color_count=6)
            #print(palette)
            #for k in palette:
                #arr_item=convert_rgb_to_css21names(k)
                #css_2_arr_pal.append(arr_item)
            #css_2_arr_pal=list(dict.fromkeys(css_2_arr_pal))
            #print(css_2_arr_pal)
            #for l in palette:
                #arr_item=convert_rgb_to_css3names(k)
                #css_3_arr_pal.append(k)
            #css_3_arr_pal=list(dict.fromkeys(css_3_arr_pal))
            #print(css_3_arr_pal)
            color=convert_rgb_to_css21names(dominant_color)
            color3=convert_rgb_to_css3names(dominant_color)
            css2.append(color)
            rgb.append(dominant_color)
            css3.append(color3)
        elif count!=0:
            color_thief = ColorThief(link)
            dominant_color = color_thief.get_color(quality=1)
            #palette = color_thief.get_palette(color_count=6)
            #print(palette)
            #for k in palette:
                #arr_item=convert_rgb_to_css21names(k)
                #css_2_arr_pal.append(arr_item)
            #css_2_arr_pal=list(dict.fromkeys(css_2_arr_pal))
            #print(css_2_arr_pal)
            #for l in palette:
                #arr_item=convert_rgb_to_css3names(l)
                #css_3_arr_pal.append(arr_item)
            #css_3_arr_pal=list(dict.fromkeys(css_3_arr_pal))
            #print(css_3_arr_pal)
            color=convert_rgb_to_css21names(dominant_color)
            color3=convert_rgb_to_css3names(dominant_color)
            css2.append(color)
            rgb.append(dominant_color)
            css3.append(color3)
        count_img+=1

    return rgb,css2,css3

def convert_rgb_to_css21names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    colors = pd.read_csv('ZoiDataColorCodes.csv')

    new_list=[]

    for i in range(0,len(colors)):
        clc = colors.loc[i]

        r1=rgb_tuple[0]
        g1=rgb_tuple[1]
        b1=rgb_tuple[2]

        r2=clc['R']
        g2=clc['G']
        b2=clc['B']
        d=abs(r1-r2)+abs(g1-g2)+abs(b1-b2)
        new_list.append(d)

    smallest=min(new_list)
    pos = new_list.index(smallest)
    return colors['Color Name'][pos]

def convert_rgb_to_css3names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

def create_zip():
    handle=zipfile.ZipFile("static/zip/label_class.zip","w")
    for i in detect_dir():
        target="static/detection/"+i
        handle.write(target,compress_type=zipfile.ZIP_DEFLATED)
    handle.close()
def create_zip2():
    handle=zipfile.ZipFile("static/zip/label_seg.zip","w")
    for i in seg_dir():
        target="static/segmentation/"+i
        handle.write(target,compress_type=zipfile.ZIP_DEFLATED)
    handle.close()
def create_zip_mask():
    handle=zipfile.ZipFile("static/zip/label_mask.zip","w")
    for i in mask_dir():
        target="static/cnn_out/"+i
        handle.write(target,compress_type=zipfile.ZIP_DEFLATED)
    handle.close()


@app.route("/",methods=["GET","POST"])
def home():
    return render_template("index.html")

@app.route("/1",methods=["POST","GET"])
def upload():
    variable=request.files.get("file")
    if variable and allowed_file(variable.filename):
        filename = secure_filename(variable.filename)
        print(filename)
        variable.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("upload.html")


@app.route("/2",methods=["POST","GET"])
def delete():
    remove_file()
    remove_class()
    remove_seg()
    remove_mask()
    return redirect("/1")

@app.route("/3",methods=["GET","POST"])
def uploaded_datas():
    dir=directory()
    if len(dir) == 0:
        return render_template("empty_uploaded.html")
    else:    
        return render_template("uploaded.html",list=dir)

@app.route("/4",methods=["GET","POST"])
def clear_uploaded():
    remove_file()
    remove_class()
    remove_seg()
    remove_mask()
    return render_template("empty_uploaded.html")

@app.route("/5/<name>")
def define_clear(name):
    dir = 'static/data/'
    os.remove(os.path.join(dir, name))
    return redirect("/3")

@app.route("/6",methods=['GET','POST'])
def clear_class():
    remove_class()
    if len(directory())==0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("predict-class.html")

@app.route("/7",methods=['GET','POST'])
def clear_seg():
    remove_seg()
    if len(directory())==0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("seg_predict.html")
@app.route("/8",methods=["GET","POST"])
def clear_mask():
    remove_mask()
    if len(directory())==0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("mask_predict.html")

@app.route("/color_predict",methods=["GET","POST"])
def predict_page():
    dir=directory()
    if len(dir) == 0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("predict-color.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    clear_seg()
    run_seg()
    dir=directory()
    dir2=seg_dir()
    rgb,css2,css3=define_main_color(dir2)
    schedule=zip(dir,rgb,css2)
    df = pd.DataFrame({'Name':dir,
    'Dominant RGB Value':rgb,'CSS2 Dominant Color Name':css2,'CSS3 Dominant Color Name':css3})

    df.to_excel('./static/excel/states.xlsx',sheet_name="States",index=False)

    return render_template("color-analysis.html",list=schedule)


@app.route("/classification_predict", methods=["GET","POST"])
def classification():
    dir=directory()
    if len(dir) == 0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("predict-class.html")

@app.route("/classification",methods=["GET","POST"])
def predict_class():
    remove_class()
    detection()
    create_zip()
    list=detect_dir()
    length=len(list)
    return render_template("classification.html",list=list,length=length)


@app.route("/segmentation_predict",methods=["GET","POST"])
def seg_predict():
    dir=directory()
    if len(dir) == 0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("seg_predict.html")

@app.route("/segmentation",methods=["GET","POST"])
def segmentation():
    remove_seg()
    run_seg()
    create_zip2()
    list=seg_dir()
    length=len(list)
    return render_template("segmentation.html",list=list,length=length)

@app.route("/label",methods=['GET','POST'])
def label():
    return render_template("label.html")

@app.route("/label_seg",methods=['GET','POST'])
def label_seg():
    list=seg_dir()
    length=len(list)
    if length==0 and len(directory())==0:
       return render_template("empty_uploaded.html")
    elif length==0 and len(directory())!=0:
        return render_template("label_alert.html") 
    else:
        return render_template("segmentation.html",list=list,length=length)

@app.route("/label_class",methods=['GET','POST'])
def label_class():
    list=detect_dir()
    length=len(list)
    if length==0 and len(directory())==0:
       return render_template("empty_uploaded.html")
    elif length==0 and len(directory())!=0:
        return render_template("label_alert.html") 
    else:
        return render_template("classification.html",list=list,length=length)

@app.route("/mask_predict",methods=["GET","POST"])
def mask_predict():
    dir=directory()
    if len(dir)==0:
        return render_template("empty_uploaded.html")
    else:
        return render_template("mask_predict.html")

@app.route("/mask_class",methods=["GET","POST"])
def mask_class():
    remove_mask()
    run_mask()
    create_zip_mask()
    list=mask_dir()
    length=len(list)
    return render_template("mask_class.html",list=list,length=length)



if __name__ =='__main__':
    app.run(host='0.0.0.0',threaded=False)