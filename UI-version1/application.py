from flask import Flask, render_template, request, redirect, jsonify
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from threading import Timer
import sys
import os
import json
import speech_recognition as sr

# Start using all the regular flask logic
flask_app = Flask(__name__)

####################################################################################
# Personal settings set up screens                                                 #
####################################################################################

# Write to main page
@flask_app.route("/", methods=["GET", "POST"])
def welcome():
    setup_done = user_info["setup_bool"]
    if request.method == "POST":
        return redirect("/name")
    elif setup_done == True:
        return redirect("/AI_bot")
    else:
        pages = calculate_pages(0)
        return render_template("welcome.html", pages=pages)

@flask_app.route("/name", methods=["GET", "POST"])
def name():
    if 'backwards' in request.form:
        # get name from form
        user_info["name"] = request.form["user_name"]
        return redirect("/")
    elif 'forwards' in request.form:
        # get name from form
        user_info["name"] = request.form["user_name"]
        return redirect("/location")
    else:
        pages = calculate_pages(1)
        user = user_info["name"]
        return render_template("name.html", pages=pages, user=user)

@flask_app.route("/location", methods=["GET", "POST"])
def location():
    if 'backwards' in request.form:
        # get location from form
        user_info["default_location"] = request.form["location"]
        return redirect("/name")
    elif 'forwards' in request.form:
        # get location from form
        user_info["default_location"] = request.form["location"]
        return redirect("/music")
    else:
        pages = calculate_pages(2)
        location = user_info["default_location"]
        return render_template("location.html", pages=pages, location=location)

@flask_app.route("/music", methods=["GET", "POST"])
def music():
    if 'backwards' in request.form:
        # get spotify username and password from form
        user_info["spotify_user"] = request.form["spot_user"]
        user_info["spotify_pass"] = request.form["spot_pass"]
        return redirect("/location")
    elif 'forwards' in request.form:
        # get spotify username and password from form
        user_info["spotify_user"] = request.form["spot_user"]
        user_info["spotify_pass"] = request.form["spot_pass"]

        # set any final user_info parameters and save it as json
        user_info["setup_bool"] = True
        with open(filepath, 'w') as save_user_info:
            json.dump(user_info, save_user_info)

        return redirect("/AI_bot")
    else:
        pages = calculate_pages(3)
        spot_user = user_info["spotify_user"]
        spot_pass = user_info["spotify_pass"]
        return render_template("music.html", pages=pages, spot_user=spot_user, spot_pass=spot_pass)


####################################################################################
# The main screen, listens to user and assists them                                #
####################################################################################

@flask_app.route("/AI_bot", methods=["GET", "POST"])
def AI_bot():
    if 'setup' in request.form:
        return redirect("/")
    elif 'start' in request.form:
        return redirect("/listen")
    else:
        return render_template("AI_bot.html")

@flask_app.route("/listen")
def listen():
    return render_template('listen.html')


@flask_app.route("/start_asr")
def start():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
    except:
        text = "I didn't quite get that please say it again"

    with open("transcript.txt", 'w+') as f:
        f.write(text)
    return jsonify("speechrecognition start success!")


@flask_app.route("/get_audio")
def get_audio():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    return jsonify(transcript)

####################################################################################
# helper functions                                                                 #
####################################################################################

# Define function for QtWebEngine
def ui(location):
    qt_app = QApplication(sys.argv)
    web = QWebEngineView()
    web.setWindowTitle("AI Assistant")
    web.resize(600, 500)
    web.setZoomFactor(1.5)
    web.load(QUrl(location))
    web.show()
    sys.exit(qt_app.exec_())

def calculate_pages(page_no):
    pages = [False]*num_pages
    pages[page_no] = True
    return pages


if __name__ == "__main__":
    # look for set-up.json if not initialise dictionary
    filepath = '../Config/set-up.json'
    num_pages = 4
    if os.path.isfile(filepath):
        with open(filepath) as setUpFile:
            user_info = json.load(setUpFile)
    else:
        user_info = {"name": "",
                     "spotify_user": "",
                     "spotify_pass": "",
                     "default_location": "",
                     "setup_bool": False}

    # start sub-thread to open the browser.
    Timer(1, lambda: ui("http://127.0.0.1:5000/")).start()
    flask_app.run(debug=False)
