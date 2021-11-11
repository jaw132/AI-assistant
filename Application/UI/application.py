from flask import Flask, render_template, request, redirect, jsonify
import webbrowser
import os
import json
import speech_recognition as sr
from win32com.client import Dispatch
import bots.UI.helpers as helpers
import time

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
    elif setup_done:
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
        user_info["setup_bool"] = False
        return redirect("/")
    else:
        return render_template("listen.html")


@flask_app.route("/bot_listen")
def start():
    print("in bot listen")
    with sr.Microphone() as source:
        audio = main_listener.listen(source)
    try:
        text = main_listener.recognize_google(audio)
    except:
        text = "!"
    print(text)

    with open("transcript.txt", 'w+') as f:
        f.write(text)
    return jsonify("speechrecognition start success!")


@flask_app.route("/get_response")
def get_audio():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    # parse function transcript and act accordingly
    parsed_transcript = parse_input(transcript)
    # feed output from parse function to action function which decides
    # action and provides response
    response = decide_action(parsed_transcript)
    return jsonify(response)

####################################################################################
# helper functions                                                                 #
####################################################################################

def parse_input(user_input):
    if "spotify" in user_input or "music" in user_input:
        return "spotify"
    elif "!" in user_input:
        return "!"
    else:
        return ""


def decide_action(parsed_text):
    if parsed_text == "spotify":
        bot_speaker.Speak("Which artist do you want to listen to")
        with sr.Microphone() as source:
            audio = function_listener.listen(source)
        try:
            artist = function_listener.recognize_google(audio)
            bot_speaker.Speak("Opening spotify and playing "+str(artist))
        except:
            bot_speaker.Speak("I couldn't recognise that artist, opening spotify with a personal choice")
            artist = "celine"

        # open spotify and play the provided user
        helpers.run_spotify(artist, user_info["spotify_user"], user_info["spotify_pass"])
        return "success"

    elif parsed_text == "!":
        return "fail"
    else:
        bot_speaker.Speak("I'm sorry, I couldn't understand that, please say it again")
        return "success"




def calculate_pages(page_no):
    pages = [False]*num_pages
    pages[page_no] = True
    return pages


if __name__ == "__main__":

    # define number of set up pages, TODO get this dynamically
    num_pages = 4

    # look for set-up.json if not initialise dictionary
    filepath = '../Config/set-up.json'
    if os.path.isfile(filepath):
        with open(filepath) as setUpFile:
            user_info = json.load(setUpFile)
    else:
        user_info = {"name": "",
                     "spotify_user": "",
                     "spotify_pass": "",
                     "default_location": "",
                     "setup_bool": False}

    # initialise the bot's listening and speaking capabilities - TODO make listener AI
    bot_speaker = Dispatch("SAPI.SpVoice")
    main_listener = sr.Recognizer()
    function_listener = sr.Recognizer()

    # run the app on local web browser
    webbrowser.open_new('http://127.0.0.1:5000/')
    flask_app.run(port=5000)
