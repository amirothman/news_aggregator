# Launch with flask

FLASK_APP=webapp.py flask run

# Launch with Gunicorn

gunicorn -b 0.0.0.0:5000 webapp:app
