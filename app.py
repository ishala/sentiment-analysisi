from flask import Flask, render_template, request
from predict import predict_comment, tokenizer
import logging

app = Flask(__name__)

# Konfigurasikan logging untuk debugging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def prediction():
    comment = request.form['comment']
    categories = ['Netral', 'Positif', 'Negatif']  # Sesuaikan dengan kategori Anda
    pred_prob, predicted_category, pred_labels = predict_comment(tokenizer, comment, categories)
    
    # Log hasil prediksi
    logging.info(f"Comment: {comment}")
    logging.info(f"Predicted Category: {predicted_category}")
    logging.info(f"Prediction Probabilities: {pred_prob}")
    logging.info(f"Prediction Labels: {pred_labels}")
    
    # Setiap kategori diberi nilai 0 atau 1
    netral = 1 if predicted_category == 'Netral' else 0
    positif = 1 if predicted_category == 'Positif' else 0
    negatif = 1 if predicted_category == 'Negatif' else 0
    
    # Tampilkan hasil prediksi
    return render_template('main.html', comment=comment, predicted_category=predicted_category, netral=netral, positif=positif, negatif=negatif)

if __name__ == '__main__':
    app.run(debug=True)



# netral = 1 if predicted_category == 0 else 0
#     positif = 1 if predicted_category == 1 else 0
#     negatif = 1 if predicted_category == 2 else 0
