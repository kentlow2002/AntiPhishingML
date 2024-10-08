from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['email']
    # Pass the user input to the result template
    return render_template('result.html', email=name)

if __name__ == '__main__':
    app.run(debug=True)