#!/usr/bin/env python3
"""
Movie Recommendation System using Collaborative Filtering
Based on MovieLens 100k dataset
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import argparse
import os

def load_data():
    """Load and prepare the MovieLens dataset"""
    print("Loading MovieLens 100k dataset...")
    
    # Load ratings data
    ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv('data/u.data', sep='\t', names=ratings_columns)
    
    # Load movie titles
    movies_columns = ['item_id', 'title', 'release_date', 'video_release', 'imdb_url', 
                     'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies_df = pd.read_csv('data/u.item', sep='|', names=movies_columns, encoding='latin-1')
    
    print(f"Dataset loaded: {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users on {ratings_df['item_id'].nunique()} movies")
    
    return ratings_df, movies_df

def explore_data(ratings_df, movies_df):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Rating distribution
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    print("Rating Distribution:")
    for rating, count in rating_counts.items():
        print(f"  Rating {rating}: {count} ratings ({count/len(ratings_df)*100:.1f}%)")
    
    # User activity
    user_activity = ratings_df['user_id'].value_counts()
    print(f"\nUser Activity:")
    print(f"  Average ratings per user: {user_activity.mean():.1f}")
    print(f"  Most active user: {user_activity.max()} ratings")
    print(f"  Least active user: {user_activity.min()} ratings")
    
    # Movie popularity
    movie_popularity = ratings_df['item_id'].value_counts()
    print(f"\nMovie Popularity:")
    print(f"  Average ratings per movie: {movie_popularity.mean():.1f}")
    print(f"  Most popular movie: {movie_popularity.max()} ratings")
    print(f"  Least popular movie: {movie_popularity.min()} ratings")

def build_model(ratings_df):
    """Build and train the collaborative filtering model"""
    print("\n=== Building Recommendation Model ===")
    
    # Define the rating scale
    reader = Reader(rating_scale=(1, 5))
    
    # Load data into Surprise dataset format
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    
    # Configure the algorithm (Item-based KNN)
    sim_options = {
        'name': 'cosine',
        'user_based': False,  # Item-based collaborative filtering
        'min_support': 3
    }
    
    algo = KNNBasic(k=40, min_k=5, sim_options=sim_options, verbose=False)
    
    # Split data and train model
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    print("Training the model...")
    algo.fit(trainset)
    
    # Evaluate the model
    print("Evaluating model performance...")
    predictions = algo.test(testset)
    accuracy = rmse(predictions, verbose=False)
    print(f"Model RMSE: {accuracy:.3f}")
    
    return algo, trainset

def get_recommendations(user_id, algo, trainset, movies_df, n=10):
    """Generate movie recommendations for a given user"""
    print(f"\nGenerating recommendations for User {user_id}...")
    
    # Get all movie IDs
    all_movie_ids = list(movies_df['item_id'])
    
    # Get movies the user has already rated
    user_ratings = []
    for item_id, rating in trainset.ur[trainset.to_inner_uid(user_id)]:
        user_ratings.append((trainset.to_raw_iid(item_id), rating))
    
    rated_movie_ids = [movie_id for movie_id, _ in user_ratings]
    
    # Find movies the user hasn't rated
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        pred = algo.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = predictions[:n]
    
    # Convert to readable format with movie titles
    recommendations_with_titles = []
    for movie_id, predicted_rating in top_recommendations:
        movie_title = movies_df[movies_df['item_id'] == movie_id]['title'].iloc[0]
        recommendations_with_titles.append({
            'movie_id': movie_id,
            'title': movie_title,
            'predicted_rating': round(predicted_rating, 2)
        })
    
    return recommendations_with_titles, user_ratings

def display_recommendations(user_id, recommendations, user_ratings, movies_df):
    """Display recommendations in a user-friendly format"""
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR USER {user_id}")
    print(f"{'='*60}")
    
    # Display user's recent ratings
    print(f"\nUser {user_id}'s recently rated movies:")
    user_ratings_with_titles = []
    for movie_id, rating in user_ratings[:5]:  # Show last 5 ratings
        movie_title = movies_df[movies_df['item_id'] == movie_id]['title'].iloc[0]
        user_ratings_with_titles.append((movie_title, rating))
    
    for title, rating in user_ratings_with_titles:
        print(f"  ★ {rating} - {title}")
    
    # Display recommendations
    print(f"\nTop {len(recommendations)} Recommended Movies:")
    print("-" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec['title']} (Predicted Rating: {rec['predicted_rating']})")

def main():
    """Main function to run the recommendation system"""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--user_id', type=int, default=196, 
                       help='User ID to generate recommendations for (default: 196)')
    parser.add_argument('--num_recommendations', type=int, default=10,
                       help='Number of recommendations to generate (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Load and explore data
        ratings_df, movies_df = load_data()
        explore_data(ratings_df, movies_df)
        
        # Build model
        algo, trainset = build_model(ratings_df)
        
        # Generate recommendations
        recommendations, user_ratings = get_recommendations(
            args.user_id, algo, trainset, movies_df, args.num_recommendations
        )
        
        # Display results
        display_recommendations(args.user_id, recommendations, user_ratings, movies_df)
        
        print(f"\n🎬 Recommendation generation complete!")
        
    except FileNotFoundError:
        print("Error: Data files not found. Please ensure the 'data/u.data' and 'data/u.item' files exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()