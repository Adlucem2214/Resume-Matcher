### Project Structure

Here's a suggested project structure:

```
resume_job_matching/
│
├── app/
│   ├── main.py                # Entry point for the FastAPI application
│   ├── models.py              # Pydantic models for request and response
│   ├── services.py            # Business logic for data processing
│   ├── similarity.py           # Similarity computation functions
│   ├── data_cleaning.py       # Data cleaning functions
│   ├── feature_engineering.py  # Feature engineering functions
│   └── utils.py               # Utility functions
│
├── tests/
│   ├── test_services.py        # Unit tests for services
│   ├── test_similarity.py       # Unit tests for similarity computations
│   └── test_data_cleaning.py   # Unit tests for data cleaning
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

### Step 1: Setting Up the Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install FastAPI and other dependencies**:
   Create a `requirements.txt` file with the following content:
   ```plaintext
   fastapi
   uvicorn
   pandas
   scikit-learn
   numpy
   pydantic
   pytest
   ```

   Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Implementing the FastAPI Application

#### `app/main.py`

```python
from fastapi import FastAPI
from models import Resume, Job
from services import match_resume_to_job

app = FastAPI()

@app.post("/match/")
async def match(resume: Resume, job: Job):
    return match_resume_to_job(resume, job)
```

#### `app/models.py`

```python
from pydantic import BaseModel

class Resume(BaseModel):
    skills: str
    experience: str
    education: str

class Job(BaseModel):
    title: str
    skills_required: str
    description: str
```

#### `app/services.py`

```python
from data_cleaning import clean_data
from feature_engineering import extract_features
from similarity import compute_similarity

def match_resume_to_job(resume, job):
    cleaned_resume = clean_data(resume)
    cleaned_job = clean_data(job)
    
    resume_features = extract_features(cleaned_resume)
    job_features = extract_features(cleaned_job)
    
    similarity_score = compute_similarity(resume_features, job_features)
    
    return {"similarity_score": similarity_score}
```

#### `app/data_cleaning.py`

```python
import pandas as pd

def clean_data(data):
    # Implement data cleaning logic here
    # For example, removing special characters, lowercasing, etc.
    return data
```

#### `app/feature_engineering.py`

```python
def extract_features(data):
    # Implement feature extraction logic here
    # For example, converting text to vectors using TF-IDF or similar
    return features
```

#### `app/similarity.py`

```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_features, job_features):
    # Compute similarity score between resume and job features
    return cosine_similarity(resume_features, job_features)[0][0]
```

### Step 3: Testing

#### `tests/test_services.py`

```python
import pytest
from app.services import match_resume_to_job

def test_match_resume_to_job():
    resume = {"skills": "Python, SQL", "experience": "2 years", "education": "B.Tech"}
    job = {"title": "Data Scientist", "skills_required": "Python, Machine Learning", "description": "Analyze data"}
    
    result = match_resume_to_job(resume, job)
    assert "similarity_score" in result
```

### Step 4: Running the Application

To run the FastAPI application, use the following command:

```bash
uvicorn app.main:app --reload
```

### Step 5: Testing the Application

To run the tests, use:

```bash
pytest tests/
```

### Conclusion

This guide provides a structured approach to creating a backend-only resume-job matching system using FastAPI. You can expand upon the data cleaning, feature engineering, and similarity computation functions based on your specific requirements and the dataset you are working with.