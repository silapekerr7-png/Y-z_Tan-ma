import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog

model = pickle.load(open('model.pkl', 'rb'))

st.title('Yüz Noktası Tespiti :smiley:')

dosya = st.file_uploader('Yüzü kadraja ortalanmış, yakın çekim bir fotoğraf yükleyin', type=['jpg', 'jpeg', 'png'])

if dosya is not None:
    resim = Image.open(dosya).convert('L').resize((96, 96))
    goruntu = np.array(resim, dtype=np.float32) / 255.
    h = hog(goruntu, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    z = model['pca'].transform([h]).astype(np.float32)
    nokta = np.concatenate([m.predict(z)[0] + o for m, o in zip(model['modeller'], model['ortalamalar'])])
    nokta = np.clip(nokta, 0, 96)

    buyuk = resim.resize((384, 384)).convert('RGB')
    cizim = ImageDraw.Draw(buyuk)
    for x, y in zip(nokta[::2], nokta[1::2]):
        cizim.ellipse((x * 4 - 4, y * 4 - 4, x * 4 + 4, y * 4 + 4), fill='red')
    st.image(buyuk, caption='Tahmin edilen 15 yüz noktası')

    st.success(f'Burun ucu: ({nokta[20]:.1f}, {nokta[21]:.1f}) | Sol göz: ({nokta[0]:.1f}, {nokta[1]:.1f}) | Sağ göz: ({nokta[2]:.1f}, {nokta[3]:.1f})')