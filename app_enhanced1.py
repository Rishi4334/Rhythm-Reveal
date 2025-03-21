from flask import Flask, render_template, request, url_for, jsonify
import os
import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import matplotlib.pyplot as plt
import base64
import json
from io import BytesIO
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (create this file with your API keys)
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize emotion detection model
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def extract_audio_features(file_path):
    """Extract comprehensive audio features"""
    try:
        y, sr = librosa.load(file_path)
        
        # Basic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        loudness = np.mean(librosa.feature.rms(y=y))
        
        # Advanced features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Calculate averages
        avg_mfcc = np.mean(mfcc, axis=1)
        avg_chroma = np.mean(chroma, axis=1)
        avg_centroid = np.mean(spectral_centroid)
        avg_bandwidth = np.mean(spectral_bandwidth)
        avg_contrast = np.mean(spectral_contrast)
        avg_rolloff = np.mean(spectral_rolloff)
        
        # Generate visualizations
        plots = generate_audio_visualizations(y, sr)
        
        return {
            'tempo': tempo,
            'loudness': loudness,
            'mfcc': avg_mfcc,
            'chroma': avg_chroma,
            'spectral_centroid': avg_centroid,
            'spectral_bandwidth': avg_bandwidth,
            'spectral_contrast': avg_contrast,
            'spectral_rolloff': avg_rolloff,
            'plots': plots
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def generate_audio_visualizations(y, sr):
    """Generate multiple audio visualizations"""
    plots = {}

    # Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plots['waveform'] = save_plot()

    # Spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plots['spectrogram'] = save_plot()

    # MFCC
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plots['mfcc'] = save_plot()

    # 🔹 **Ensure chroma plot is always generated**
    plt.figure(figsize=(10, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plots['chroma'] = save_plot()

    return plots

def save_plot():
    """Save current plot to base64 string"""
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_data

def analyze_lyrics(lyrics, audio_features=None, use_api=False):
    """Perform comprehensive lyrics analysis"""
    try:
        # Try to truncate lyrics if they exceed the maximum token limit
        try:
            max_tokens = 512
            if len(lyrics.split()) > max_tokens:
                lyrics = ' '.join(lyrics.split()[:max_tokens])  # Keep only the first 512 tokens
        except Exception as e:
            print(f"Error while truncating lyrics: {e}")
            return None  # If truncation fails, stop processing

        # Original analysis
        blob = TextBlob(lyrics)

        # Sentiment analysis
        sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

        # Topic modeling
        topics = extract_topics(lyrics)

        # Emotion analysis
        emotions = emotion_pipeline(lyrics)

        # Enhanced analysis with OpenAI - only if explicitly requested
        if use_api:
            try:
                ai_analysis = get_ai_lyrics_analysis(lyrics, audio_features)
            except Exception as e:
                print(f"Error in OpenAI analysis, falling back to basic analysis: {e}")
                ai_analysis = {
                    'themes': ["Analysis unavailable"],
                    'storytelling': "API analysis unavailable",
                    'writing_style': "N/A",
                    'cultural_references': []
                }
        else:
            ai_analysis = {
                'themes': ["Manual analysis only"],
                'storytelling': "API analysis not requested",
                'writing_style': "N/A",
                'cultural_references': []
            }

        # Generate summary - use API only if requested
        if use_api:
            try:
                summary = generate_ai_summary(lyrics, sentiment, emotions, topics, audio_features, ai_analysis)
            except Exception as e:
                print(f"Error in OpenAI summary, falling back to basic summary: {e}")
                summary = generate_fallback_summary(sentiment, emotions, topics)
        else:
            summary = generate_fallback_summary(sentiment, emotions, topics)

        return {
            'sentiment': sentiment,
            'topics': topics,
            'emotions': emotions,
            'ai_analysis': ai_analysis,
            'summary': summary
        }

    except Exception as e:
        print(f"Error analyzing lyrics: {e}")
        return None

def get_ai_lyrics_analysis(lyrics, audio_features=None):
    """Use OpenAI to analyze lyrics more deeply"""
    if not os.getenv('OPENAI_API_KEY'):
        raise Exception("OpenAI API key not found")

    try:
        if not lyrics.strip():
            return {
                'themes': ["No lyrics provided"],
                'storytelling': "No lyrics to analyze",
                'writing_style': "N/A",
                'cultural_references': []
            }

        # Prepare context about audio features if available
        audio_context = f"The song has a tempo of {audio_features['tempo']} BPM." if audio_features else ""

        prompt = f"""
        Analyze these lyrics comprehensively:

        {lyrics}

        {audio_context}

        Provide the following:
        1. Main themes (list of 3-5 themes)
        2. Storytelling assessment (is there a narrative, what's the perspective)
        3. Writing style (descriptive, metaphorical, direct, etc.)
        4. Any cultural or historical references

        Format as JSON with keys: themes, storytelling, writing_style, cultural_references
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a music and lyrics analysis expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        import json
        try:
            ai_analysis = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            ai_analysis = {
                'themes': ["Could not parse response"],
                'storytelling': response.choices[0].message.content[:200] + "...",
                'writing_style': "Unknown",
                'cultural_references': []
            }

        # 🔹 **Ensure all expected fields are always present**
        if 'themes' not in ai_analysis:
            ai_analysis['themes'] = ["N/A"]
        if 'writing_style' not in ai_analysis:
            ai_analysis['writing_style'] = "N/A"
        if 'storytelling' not in ai_analysis:
            ai_analysis['storytelling'] = "N/A"
        if 'cultural_references' not in ai_analysis:
            ai_analysis['cultural_references'] = []

        return ai_analysis

    except Exception as e:
        print(f"Error in AI lyrics analysis: {e}")
        return {
            'themes': ["Analysis failed"],
            'storytelling': "N/A",
            'writing_style': "N/A",
            'cultural_references': []
        }

def generate_ai_summary(lyrics, sentiment, emotions, topics, audio_features, ai_analysis):
    """Generate a comprehensive and meaningful song summary using OpenAI"""
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        raise Exception("OpenAI API key not found")
        
    try:
        # If no lyrics, generate summary based only on audio
        if not lyrics or lyrics.strip() == "":
            if audio_features:
                # Create a prompt focused on audio features
                audio_prompt = f"""
                Create a brief summary of an instrumental track with these characteristics:
                - Tempo: {audio_features['tempo']} BPM
                - Spectral features indicating the tone color and texture
                
                Describe what emotions and moods this instrumental likely evokes based on these features.
                Keep the response under 150 words and make it engaging.
                """
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a music analysis expert."},
                        {"role": "user", "content": audio_prompt}
                    ],
                    max_tokens=250
                )
                
                return response.choices[0].message.content
            else:
                return "No audio or lyrics data available for analysis."
        
        # Extract main emotion and confidence
        main_emotion = emotions[0]['label']
        emotion_score = emotions[0]['score']
        
        # Build prompt with all available data
        prompt = f"""
        Create a compelling summary of a song with these analyses:
        
        LYRICS FRAGMENT: "{lyrics[:150]}..."
        
        SENTIMENT: Polarity ({sentiment['polarity']:.2f}), Subjectivity ({sentiment['subjectivity']:.2f})
        
        EMOTION: {main_emotion} (confidence: {emotion_score:.2f})
        
        TEMPO: {audio_features['tempo'] if audio_features else 'Unknown'} BPM
        
        TOPICS: {', '.join(topics[:2])}
        
        AI ANALYSIS:
        - Themes: {', '.join(ai_analysis.get('themes', ['Unknown'])[:3])}
        - Style: {ai_analysis.get('writing_style', 'Unknown')}
        
        Create an engaging 2-3 sentence summary that captures the essence of this song, its likely impact on listeners, and its artistic value.
        """
        
        # Call OpenAI API
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a music critic and analyst with expertise in providing insightful song summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        raise

def generate_fallback_summary(sentiment, emotions, topics):
    """Generate a basic summary without OpenAI (fallback)"""
    try:
        # Extract main emotion and confidence
        main_emotion = emotions[0]['label']
        emotion_score = emotions[0]['score']
        
        # Analyze sentiment
        sentiment_tone = "neutral"
        if sentiment['polarity'] > 0.2:
            sentiment_tone = "positive"
        elif sentiment['polarity'] < -0.2:
            sentiment_tone = "negative"
        
        # Build summary
        summary = f"This song carries a {sentiment_tone} tone with a strong sense of {main_emotion} (confidence: {emotion_score:.2f}). "
        if topics:
            topic_terms = []
            for topic in topics[:2]:  # Use first two topics
                if ": " in topic:
                    terms = topic.split(": ")[1].split(" + ")
                    topic_terms.extend(terms)
            if topic_terms:
                summary += f"It explores themes like {', '.join(topic_terms[:4])}. "
        
        summary += f"The lyrics suggest a {sentiment_tone} narrative with emotional depth."
        
        return summary
    except Exception as e:
        print(f"Error generating fallback summary: {e}")
        return "Unable to generate a detailed summary."

def extract_topics(lyrics):
    """Extract topics from lyrics"""
    try:
        # Skip if lyrics are empty
        if not lyrics or lyrics.strip() == "":
            return []
            
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([lyrics])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)
        
        topics = []
        for idx, topic in enumerate(lda.components_):
            feature_names = vectorizer.get_feature_names_out()
            topic_terms = [feature_names[i] for i in topic.argsort()[-4:]]
            topics.append(f"Topic {idx}: " + " + ".join(topic_terms))
        
        return topics
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []

@app.route('/')
def index():
    return render_template('index_enhanced2.html')

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics():
    """API endpoint for generating lyrics based on parameters"""
    try:
        # Get parameters from request
        track_name = request.form.get('track_name', '')
        artist = request.form.get('artist', '')
        mood = request.form.get('mood', '')
        genre = request.form.get('genre', '')
        
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'message': 'OpenAI API key not found. Please add your key to the .env file.',
                'lyrics': ''
            })
        
        # Create prompt for OpenAI
        prompt = f"""
        Write song lyrics for a {genre} song titled "{track_name}" by {artist}.
        The song should have a {mood} mood.
        Include verses and a chorus, and make it sound authentic to the genre.
        Keep it to a reasonable length for a typical song (around 20-30 lines).
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled songwriter who can write lyrics in any genre."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        # Get lyrics from response
        lyrics = response.choices[0].message.content
        
        return jsonify({
            'success': True,
            'message': 'Lyrics generated successfully!',
            'lyrics': lyrics
        })
        
    except Exception as e:
        print(f"Error generating lyrics: {e}")
        return jsonify({
            'success': False,
            'message': f'Error generating lyrics: {str(e)}',
            'lyrics': ''
        })

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return "No file part"
    
    file = request.files['audio']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract audio features
        audio_analysis = extract_audio_features(file_path)
        
        # Get lyrics from the form
        lyrics = request.form.get('lyrics', '')

        # Debugging: Print the length and content of the input lyrics
        print(f"Input Lyrics Length: {len(lyrics.split())}")
        print(f"Input Lyrics: {lyrics[:1000]}")  # Print first 1000 characters for inspection

        # Check if API usage is requested
        use_api = request.form.get('use_api', 'false').lower() == 'true'

        # Analyze lyrics (Fix: Define `lyrics_analysis` before using it)
        lyrics_analysis = analyze_lyrics(lyrics, audio_analysis, use_api)

        # Handle analysis errors
        if lyrics_analysis is None:
            return render_template('error.html', message="Error analyzing lyrics. Please check your input.")

        # Render results
        return render_template('results_enhanced2.html',
                               filename=filename,
                               audio_analysis=audio_analysis,
                               lyrics_analysis=lyrics_analysis,
                               song_summary=lyrics_analysis['summary'],
                               api_used=use_api)
    
if __name__ == '__main__':
    app.run(debug=True)
