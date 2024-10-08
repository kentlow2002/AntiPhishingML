from flask import Flask, render_template, request
from model import tester

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/submit', methods=['POST'])
def submit():
    email_body = request.form['email-body']
    sender_domain = request.form['sender-domain']
    # Pass the user input to the result template
    prediction = tester(email_body, sender_domain)
    return render_template('result.html', email=prediction)

if __name__ == '__main__':
    app.run(debug=True)