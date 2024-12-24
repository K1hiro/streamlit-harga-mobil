import pickle
import streamlit as st 


model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title('Estimasi Harga Mobil Bekas')

year = st.number_input('Input Tahun Keluaran Mobil')

mileage = st.number_input('Input KM Mobil')

tax = st.number_input('Input Pajak Mobil')

mpg = st.number_input('Input Konsumsi BBM Mobil')

engineSize = st.number_input('Input Kapasitas Mesin')

if st.button('Estimasi Harga'):
    estimasi = model.predict(
        [[year, mileage, tax, mpg, engineSize]]
    )
    st.write('Estimasi harga mobil bekas dalam pounds :', estimasi)
    st.write('Estimasi harga mobil bekas dalam pounds :', estimasi*16000)