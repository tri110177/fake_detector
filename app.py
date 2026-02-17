import streamlit as st
import pickle
import numpy as np
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


st.sidebar.markdown("""
    Disusun Oleh:   
    Tri Wahyu Setiawan             
    """)


st.subheader ('Welcome to Homepage')
st.markdown("""
    Selamat datang di aplikasi deteksi berita hoax!

    Aplikasi ini dirancang untuk membantu Anda **memverifikasi keaslian berita** yang Anda temukan secara online. Dengan memanfaatkan model *machine learning* canggih (Passive Aggressive Classifier), kami dapat menganalisis teks berita dan memprediksinya sebagai **REAL** atau **FAKE**.

    **Cara Penggunaan:**
    1.  Masukkan teks berita yang ingin Anda periksa ke dalam kotak teks yang tersedia.
    2.  Klik tombol "Prediksi Berita".  
    3.  Hasilnya akan muncul di bagian bawah apakah berita tersebut **REAL NEWS** atau **FAKE NEWS.**    

    Kami berharap aplikasi ini dapat membantu Anda menjadi konsumen berita yang lebih cerdas dan kritis.
        """)
st.write("Jika Anda memiliki pertanyaan, silakan hubungi kami")

st.markdown("---") 
        
data_baru = st.text_area('Masukkan berita yang akan diperiksa')

#clean text
def clean (Text) :
  sms = re.sub('[^a-zA-Z]', ' ', Text) #menghilangkan semua yang non abjad dibatasi dengan spasi
  sms = sms.lower() #mengganti ke huruf kecil semua
  sms = sms.split()
  sms = ' '.join(sms)
  if not sms.strip():
    return ""
  return sms

#text tokenize
def tokenize(text):
    return nltk.word_tokenize(text)
  
#stopwords
nltk.download('stopwords')
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]

#text lematization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatize(tokens):
    return [lemmatizer.lemmatize(word, pos='v') for word in tokens]
  
processed = []
for text in data_baru:
    cleaned = clean(text)
    tokens = tokenize(cleaned)
    nostop = remove_stopwords(tokens)
    lemma = lemmatize(nostop)
    final_text = " ".join(lemma)
    processed.append(final_text)
    
tfidf = TfidfVectorizer()
label_encoder = LabelEncoder()
model_pac = PassiveAggressiveClassifier(max_iter=10)
    
# Load model dan vectorizer
tfidf = pickle.load(open("tf_idf_vectorizer.pkl", "rb"))
model_pac = pickle.load(open("model_pac.pkl", "rb"))

#Fungsi prediksi

def prediksi(text) :  
    final_text = processed(text)
    if final_text == '':
        return None

    X_new = tfidf.transform([final_text])
    y_pred = model_pac.predict(X_new) 
    return y_pred[0]


#menampilkan hasil

st.button("Cek Berita")
    
    
    
                     







