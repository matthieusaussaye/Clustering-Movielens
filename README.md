Docker commands
=======

Build :
$ docker-compose build


Run :

Flask commands
=======

API Queries :

- In a python terminal :

$ export FLASK_APP=flask_movie_reco.py

$ flask run

Train / Vizualise the model : 
=======

Machine Learning Web interface :

-  Train the model :
http://127.0.0.1:5000/training :

- Querying an embedding of a user :
http://127.0.0.1:5000/querying
- Enter the 'user_id' you want to see the associated cluster.
- See the heatmap with 60 Top Rated movies.


Use the notebook "K-Means Clustering of movies.ipynb" :
=======

- Access the jupyter interface :
http://127.0.0.1:8888/

- Open "K-Means Clustering of movies.ipynb"

