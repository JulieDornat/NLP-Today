
# NLP Today
## NLP Today - TP1

This project uses Python 3.12 with spaCy and scispaCy for NLP tasks.

### Requirements

- Python 3.12
- A virtual environment

### Setup

1. Create and activate a virtual environment.
2. Install the required packages:

```bash
pip install spacy==3.7.4 scispacy==0.6.2
```

### Models

Install the required language models:

```bash
python -m spacy download en_core_web_sm
pip install --upgrade https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```
or 

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz
```

### Notes

- The `en_core_web_sm` model is for general English spaCy processing.
- The `en_core_sci_sm` model is for biomedical/scientific text processing.
