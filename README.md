# Music Analysis and Lyric Generation Application

## Overview

This project is a web-based application designed to analyze audio files and lyrics, providing users with insights into the emotional and thematic elements of songs. The application leverages advanced technologies, including audio feature extraction, natural language processing, and machine learning, to deliver a comprehensive analysis of music and lyrics.

## Features

- **Audio Feature Extraction**: Analyze audio files to extract key features such as tempo, loudness, MFCC, and spectral characteristics.
- **Lyrics Analysis**: Perform sentiment analysis, topic modeling, and emotion detection on user-provided lyrics.
- **AI-Enhanced Insights**: Generate summaries and original lyrics using OpenAI's GPT-3.5-turbo model.
- **User-Friendly Interface**: A clean and intuitive web interface for seamless interaction.

## Technologies Used

- **Flask**: A lightweight web framework for Python.
- **Librosa**: A library for music and audio analysis.
- **NumPy**: A package for scientific computing in Python.
- **Matplotlib**: A plotting library for visualizing audio data.
- **TextBlob**: A library for processing textual data and performing sentiment analysis.
- **Scikit-learn**: A machine learning library for data analysis and topic modeling.
- **Transformers**: A library for natural language processing tasks.
- **OpenAI API**: For advanced lyric analysis and generation.
- **Werkzeug**: A WSGI utility library for handling file uploads.
- **dotenv**: For managing environment variables.

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
```

### Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

### Run the Application
```bash
python app_enhanced1.py
```

### Access the Application
Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage

1. **Upload an Audio File**: Click on the upload button to select an audio file from your device.
2. **Input Lyrics**: Enter the lyrics you want to analyze in the provided text area.
3. **Analyze**: Click the "Analyze" button to process the audio and lyrics.
4. **View Results**: The results page will display the audio features, lyrics analysis, and any generated summaries or lyrics.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Special thanks to the developers of the libraries and tools used in this project.
- Thanks to OpenAI for providing the GPT-3.5-turbo model for lyric generation.

## Contact

For any inquiries or feedback, please contact [231fa04334@gmail.com].

