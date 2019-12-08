# LACE

## Installation

```shell
mkdir -p ~/venv-environments/lace
python3 -m venv ~/venv-environments/lace
source ~/venv-environments/lace/bin/activate
pip install flask numpy pandas scipy snapshottest scikit-learn matplotlib seaborn
pip install --no-binary orange3 orange3==3.15.0
```

## Running the demo

```shell
FLASK_ENV=development flask run
```

```shell
cd demo_frontend/my-app
npm install # Only needed on the first run
npm start
```
