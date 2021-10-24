from flask import Flask, session
from flask_session import Session
from clustering_kmeans import ClusteringMovies
from flask import render_template
from flask import request


app = Flask(__name__)
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route("/training")
def train() :
    """Train the model"""
    clustering = ClusteringMovies()
    session['model']= clustering.kmeans_modelling()
    session['clustering_class'] = clustering

    return f"Training the model : Model predictions {session['model']}"


@app.route('/querying')
def my_form():
    """Render the form"""
    if 'clustering_class' in session:
        return render_template('my-form.html')
    else :
        return "Train the model before querying"


@app.route("/querying", methods=['POST'])
def query() :
    """Plot the query"""
    user_id = int(request.form['text'])
    if 'clustering_class' in session:
        clustering = session['clustering_class']
        user_embedding_file = clustering.user_embedding(user_id=user_id,max_users=70,max_movies=50)
        return render_template(user_embedding_file)
    else :
        return "Train the model before querying"

#if __name__ == "__main__":
#    app.run(host ='0.0.0.0', port = 5000, debug = True) 