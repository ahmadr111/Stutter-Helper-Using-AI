from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, validators
import os
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import StandardScaler
import subprocess
import whisper
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from text_to_speech import save
from auto_tune import autotune
from auto_tune import aclosest_pitch_from_scale
from auto_tune import closest_pitch
from auto_tune import degrees_from
from flask import jsonify
import soundfile as sf
import scipy.signal as sig
import psola
from functools import partial
from pathlib import Path
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import shutil


# Configuration constants
MODEL_PATH = 'model.pkl'
UPLOAD_DIR = 'uploads'

# Load the machine learning model from the pickle file
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your-secret-key'

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'fyp'
app.config['UPLOAD_DIR'] = UPLOAD_DIR
mysql = MySQL(app)

# Ensure the upload directory exists
if not os.path.exists(app.config["UPLOAD_DIR"]):
    os.makedirs(app.config["UPLOAD_DIR"])


def enhance_audio(audio_data, correction_method='closest', plot=False, scale=None):
    """
    Enhances audio data using auto-tuning.

    Args:
        audio_data (bytes): The raw audio data content.
        correction_method (str, optional): The pitch correction method to use.
            Defaults to 'closest'. Valid options are 'closest' and 'scale'.
        plot (bool, optional): If True, a plot of the results will be generated.
            Defaults to False.
        scale (str, optional): The musical scale to use for pitch correction with the
            'scale' method. Only used if correction_method is 'scale'.

    Returns:
        numpy.ndarray: The enhanced audio data as a NumPy array.
    """

    # Load audio data from memory
    y, sr = librosa.load(audio_data, sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Pick the pitch adjustment strategy
    correction_function = closest_pitch if correction_method == 'closest' else (
        partial(aclosest_pitch_from_scale, scale=scale) if scale else None
    )

    # Perform auto-tuning
    pitch_corrected_y = autotune(y, sr, correction_function, plot)
    pitch_corrected_y = pitch_corrected_y.astype(np.float32)

    return pitch_corrected_y

def audio_to_text(audio):
    # audio = INHANCEMENT(audio)
    model = whisper.load_model("base")
    # y, sr = librosa.load(audio, sr=None)  # Load audio from memory
    result = model.transcribe(audio)
    return result["text"]

def TEXT_CORRECTION(text):
    """
  Corrects words using TextBlob, removes duplicate sequential words, improves fluency, and adds punctuation.

  Args:
      text (str): The input sentence to be processed.

  Returns:
      str: The corrected, deduplicated, improved, and punctuated text.
  """

  # Download resources for NLTK (may need first-time download)
    nltk.download('punkt')

    # TextBlob correction
    corrected_text = TextBlob(text).correct()

    # Convert corrected text back to string
    corrected_text = str(corrected_text)

    # Remove extra punctuation
    cleaned_text = []
    for char in corrected_text:
        if char.isalnum() or char.isspace() or char in ("'", "?"):  # Keep alphanumeric, space, apostrophe, and question mark
            cleaned_text.append(char)
    cleaned_text = "".join(cleaned_text)

    # Sentence splitting (using NLTK sentence tokenizer)
    sentences = nltk.sent_tokenize(cleaned_text)

    # Improve fluency within and across sentences
    improved_sentences = []
    prev_words = set()  # Track previously encountered words across sentences
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower())
        deduplicated_words = []
        for word in tokens:
            if word not in prev_words or word.isupper():  # Keep first occurrence or uppercase words
                deduplicated_words.append(word)
                prev_words.add(word)
        improved_text = " ".join(deduplicated_words)
        improved_sentences.append(improved_text.title() + ".")
    prev_words.clear()  # Reset for the next iteration

    # Join improved sentences
    improved_text = " ".join(improved_sentences)

    return improved_text


def text_to_audio(text, username):
    language = "en"
    output_file = f'regenrate_input_{username}.mp3'
    save(text, language, file=output_file)
    return output_file

# def text_to_audio(text):
#     language = "en"
#     output_file = "C:/Users/Home/Desktop/FYP_PROJECT/regenerate_audio/regenrate_input.mp3"

#     save(text, language, file=output_file)
#     return output_file

def preprocess_audio(audio_file):
    audio_data, sr = librosa.load(audio_file)
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    desired_length = 1000

    zero_crossing = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    if len(zero_crossing) < desired_length:
        zero_crossing = np.pad(zero_crossing, (0, desired_length - len(zero_crossing)), mode='constant')
    elif len(zero_crossing) > desired_length:
        zero_crossing = zero_crossing[:desired_length]

    zero_crossing_arr = np.array(zero_crossing)
    features_array = np.vstack(zero_crossing_arr)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_array)
    zero_crossing_arr_normalised_input = np.array(normalized_features.tolist())
    return zero_crossing_arr_normalised_input

def predict_stutter(audio_file):
    features = preprocess_audio(audio_file)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction

# def Enhance_Audio(audio_path):
#     audio_path_array = enhance_audio(audio_path)
#     text = audio_to_text(audio_path_array)
#     corrected_text = TEXT_CORRECTION(text)
#     final_audio_path = text_to_audio(corrected_text)
#     return final_audio_path

def Enhance_Audio(audio_path, username):
    audio_path_array = enhance_audio(audio_path)
    text = audio_to_text(audio_path_array)
    corrected_text = TEXT_CORRECTION(text)
    final_audio_path = text_to_audio(corrected_text, username)
    return final_audio_path,corrected_text,text


@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        session['username'] = username

        upload_dir = app.config['UPLOAD_DIR']
        user_upload_dir = os.path.join(upload_dir, username)
        if os.path.exists(user_upload_dir):
            for file_name in os.listdir(user_upload_dir):
                file_path = os.path.join(user_upload_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        pwd = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO login (username, password) VALUES (%s, %s)", (username, pwd))
        mysql.connection.commit()
        cur.close()
        session['username'] = username
        flash('Registration successful! You can now log in.')
        return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = session['username']
        user_upload_dir = os.path.join(UPLOAD_DIR, username)
        if not os.path.exists(user_upload_dir):
            os.makedirs(user_upload_dir)
        
        uploaded_audios_dir = os.path.join(user_upload_dir, 'uploaded_audios')
        if not os.path.exists(uploaded_audios_dir):
            os.makedirs(uploaded_audios_dir)

        initial_predictions = []
        audio_texts = {}

        for file in request.files.getlist('file'):
            file_path = os.path.join(uploaded_audios_dir, file.filename)
            file.save(file_path)
            try:
                initial_prediction = predict_stutter(file_path)
                initial_predictions.append((file.filename, initial_prediction))
                if initial_prediction == 1:
                    flash(f'{file.filename} is predicted to be a stuttering audio.')
                else:
                    flash(f'{file.filename} is predicted not to be a stuttering audio.')

            except Exception as e:
                flash(f'Error processing {file.filename}: {str(e)}')

        # Retrieve uploaded files after saving
        user_audios_dir = os.path.join(UPLOAD_DIR, session['username'], 'uploaded_audios')
        files = os.listdir(user_audios_dir) if os.path.exists(user_audios_dir) else []
        
        return render_template('upload.html', files=files, initial_predictions=initial_predictions)

    # GET request: Show uploaded files if any
    user_audios_dir = os.path.join(UPLOAD_DIR, session['username'], 'uploaded_audios')
    files = os.listdir(user_audios_dir) if os.path.exists(user_audios_dir) else []
    return render_template('upload.html', files=files)


def audio_to_text_initial(audio):
    # audio = INHANCEMENT(audio)
    model = whisper.load_model("base")
    y, sr = librosa.load(audio, sr=None)  # Load audio from memory
    result = model.transcribe(y)
    return result["text"]



@app.route('/convert_text', methods=['POST'])
def convert_text():
    username = session['username']
    user_audios_dir = os.path.join(UPLOAD_DIR, username, 'uploaded_audios')
    files = os.listdir(user_audios_dir) if os.path.exists(user_audios_dir) else []
    audio_texts = {}

    for file in files:
        file_path = os.path.join(user_audios_dir, file)
        print(f"Converting file: {file_path}")  # Print the file path for debugging
        if os.path.exists(file_path) and os.access(file_path, os.R_OK):
            try:
                text = audio_to_text_initial(file_path)
                audio_texts[file] = text
            except Exception as e:
                flash(f'Error converting {file} to text: {str(e)}')
        else:
            flash(f"File not found: {file_path}")

    return render_template('upload.html', files=files, audio_texts=audio_texts)

# @app.route('/process_results', methods=['POST'])
# def process_results():
#     UPLOAD_DIR = 'regenerate_audio'
#     username = session['username']
#     file_path = os.path.join('regenerate_audio', username, 'regenrate_input.mp3')

#     # Predict stuttering for final audio
#     final_prediction = predict_stutter(file_path)

#     result_data = [{
#         'filename': 'regenrate_input.mp3',
#         'final_audio_path': file_path,
#         'final_prediction': final_prediction
#     }]

#     return render_template('results.html', result_data=result_data)

@app.route('/process_results', methods=['POST'])
def process_results():
    username = session['username']
    user_upload_dir = os.path.join(UPLOAD_DIR, username)
    uploaded_audios_dir = os.path.join(user_upload_dir, 'uploaded_audios')
    if not os.path.exists(uploaded_audios_dir):
            os.makedirs(uploaded_audios_dir)
    # uploaded_audios_dir = os.path.join(user_upload_dir, 'uploaded_audios')
    

    result_data = []

    for file_name in os.listdir(uploaded_audios_dir):
        file_path = os.path.join(uploaded_audios_dir, file_name)
        final_audio_enhance_path,corrected_text,text = Enhance_Audio(file_path, username)
        print(f"Converting file: {final_audio_enhance_path}")
        source_file = f'regenrate_input{username}.mp3'
        destination_file = os.path.join(app.static_folder, 'regenerate_audio', f'regenrate_input_{username}.mp3')

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)

        try:
            # Copy the file to the destination directory
            shutil.copy2(source_file, destination_file)
        except Exception as e:
            print(f"Failed to copy {source_file} to {destination_file}. Reason: {e}")


        # Predict stuttering for final audio
        final_prediction = predict_stutter(final_audio_enhance_path)
        print(f"Prediction: {final_prediction}")
       

        result_data.append({
            'filename': file_name,
            'final_audio_path': final_audio_enhance_path,
            'final_prediction': final_prediction,
            'text':text,
            'corrected_text': corrected_text
            
        })


     # Render the results directly in the process_results function

    return render_template('results.html', result_data=result_data)

    

@app.route('/uploaded_audios/<filename>')
def uploaded_file(filename):
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    user_audios_dir = os.path.join(app.config['UPLOAD_DIR'], session['username'], 'uploaded_audios')

    # Check if file exists before sending
    if not os.path.exists(os.path.join(user_audios_dir, filename)):
        flash(f'File "{filename}" not found.')
        return redirect(url_for('upload'))

    return send_from_directory(user_audios_dir, filename)


# @app.route('/uploaded_audios/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_DIR'], filename)
class StutterForm(Form):
    name = StringField('Name', [validators.Length(min=4, max=25)])
    age = StringField('Age', [validators.Length(min=1, max=3)])
    contact_info = StringField('Contact Information (email/phone number)', [validators.Length(min=6, max=35)])
    stutter_background = TextAreaField('Stuttering Background', [validators.Length(min=10)])
    stutter_patterns = TextAreaField('Stuttering Patterns', [validators.Length(min=10)])
    communication_style = TextAreaField('Communication Style', [validators.Length(min=10)])
    goals_expectations = TextAreaField('Goals and Expectations', [validators.Length(min=10)])
    additional_comments = TextAreaField('Additional Comments', [validators.Length(min=10)])

@app.route('/stutter_form', methods=['GET', 'POST'])
def stutter_form():
    form = StutterForm(request.form)
    if request.method == 'POST' and form.validate():
        # Process the form data here
        # For example, you can store the data in a database or send it to an email
        return 'Form submitted successfully!'
    return render_template('stutter_form.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
