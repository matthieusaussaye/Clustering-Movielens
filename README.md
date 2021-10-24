Docker commands
=======

- Build : `$ docker-compose build`
- Run : `$ docker-compose up`


Flask commands
=======
Case if you dont want to use dockerfile.

API Queries / In a python terminal : 

- Create flask app : `$ export FLASK_APP=flask_movie_reco.py`
- Deploy flask app : `$ flask run`

Train / Vizualise the model : 
=======

Machine Learning Web interface :

-  Train the model :
http://127.0.0.1:5000/training :

- Querying an embedding of a user :
http://127.0.0.1:5000/querying
- Enter the 'user_id' you want to see the associated cluster.
- It automaticaly plot he heatmap with 60 Top Rated movies of the group of the user.


Use the notebook "K-Means Clustering of movies.ipynb" :
=======

- Access the jupyter interface :
http://127.0.0.1:8888/

- Open "K-Means Clustering of movies.ipynb"

