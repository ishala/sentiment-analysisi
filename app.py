from flask import Flask, render_template, request
from predict import predict_comment, tokenizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def prediction():
    comment = request.form['comment']
    categories = ['Netral', 'Positif', 'Negatif']  # Sesuaikan dengan kategori Anda
    pred_prob, predicted_category, pred_labels = predict_comment(tokenizer, comment, categories)
    
    # Setiap kategori diberi nilai 0 atau 1
    netral = 1 if predicted_category == 0 else 0
    positif = 1 if predicted_category == 1 else 0
    negatif = 1 if predicted_category == 2 else 0
    
    # Tampilkan hasil prediksi
    return render_template('main.html', comment=comment, predicted_category=predicted_category, netral=netral, positif=positif, negatif=negatif)

if __name__ == '__main__':
    app.run(debug=True)

