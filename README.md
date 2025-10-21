# Movie Recommendation System 🎬

This project is a content-based movie recommendation system that suggests films similar to a user's choice. By analyzing movie metadata, the system identifies and ranks movies that share similar characteristics, providing personalized recommendations.

The recommendation engine is powered by machine learning techniques and exposed through a clean RESTful API built with FastAPI.

## Features

Content-Based Filtering: Recommends movies by calculating the similarity between them based on their plot, genres, keywords, and other metadata.​

Vector-Based Similarity: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert movie text data into a matrix of feature vectors.

Cosine Similarity: Employs cosine similarity to determine the likeness between movies in the vector space, ensuring relevant recommendations.

REST API: A high-performance API built with FastAPI to serve movie recommendations efficiently.​

Interactive API Docs: Automatically generated, interactive API documentation available through Swagger UI and ReDoc.

## Tech Stack

Backend: Python, FastAPI

Machine Learning: Scikit-learn, Pandas, NumPy

Web Server: Uvicorn

Data Source: The Movie Database (TMDB) 5000 Movie Dataset (or similar)

## How It Works

The recommendation logic is straightforward yet powerful:

Data Preprocessing: The system merges and cleans movie metadata (e.g., overview, genres, keywords, cast, crew).

Feature Engineering: A "tags" column is created by combining all relevant textual data for each movie.

Vectorization: The TfidfVectorizer from Scikit-learn converts the text tags into a numerical vector for each movie.

Similarity Calculation: The cosine similarity is computed between the vector of the user's chosen movie and all other movies in the dataset.

Recommendation: The system returns a list of the top 5 or 10 movies with the highest similarity scores.

##  Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites
```
Python 3.9+
Git
```
### Installation & Setup

### Clone the repository:
```
bash
git clone https://github.com/anas-fareedi/movie_recommandation.git
cd movie_recommandation
```

### Create and activate a virtual environment:
```
bash
# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### Install the necessary dependencies:
```
bash
pip install -r requirements.txt
```
### Prepare the Dataset:

Make sure you have the required dataset files (e.g., tmdb_5000_movies.csv, tmdb_5000_credits.csv) in a data/ directory within the project.

### Run the Application:
Start the FastAPI server using Uvicorn.
```
bash
uvicorn main:app --reload
The API will be live at http://127.0.0.1:8000.
```
 ### API Documentation
 
Once the server is running, you can access the interactive API documentation at:
```
Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc
```
You can test the recommendation endpoint by providing a movie title.

 How to Contribute
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.
Fork the Project.

Create your Feature Branch (git checkout -b feature/YourAmazingFeature).
Commit your Changes (git commit -m 'Add some AmazingFeature').
Push to the Branch (git push origin feature/AmazingFeature).

Open a Pull Request.
