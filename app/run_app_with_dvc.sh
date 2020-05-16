# On production the model artifacts won't be present as these reside in DVC, not the Git repo.
# This script pulls down the train.dvc stage outputs prior to running the app.
git init # Needed as dvc pull will not work if this isn't a Git repo.
dvc pull train.dvc
gunicorn app.main:app
