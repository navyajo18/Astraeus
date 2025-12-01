from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'message': 'temp'})

if __name__ == '__main__':
    app.run(debug=True)