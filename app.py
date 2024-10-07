from flask import Flask, render_template

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    user = "John Doe"
    return render_template('index.html', username=user)

@app.route('/about')
def about():
    return render_template('about.html')