# cil-twitter

## Installing dependencies

Set up a virtual environment for python3:

```bash
pip3 install virtualenv --user
virtualenv cil-env
source cil-env/bin/activate
```
To install required packages, run:

```bash
pip3 install -r requirements.txt
```
You may need to use the `--user` flag for installation of packages.

## Downloading data files
To download all required files, run:

```bash
chmod +x download.sh
./download.sh
```

## Training models & Validating models

Move to the scripts directory:
```bash
cd scripts
```

To train all models, run:
```bash
chmod +x train_all_models.sh
./train_all_models.sh
```

To validate all pre-trained models, run:
```bash
chmod +x eval_all_models.sh
./eval_all_models.sh
```