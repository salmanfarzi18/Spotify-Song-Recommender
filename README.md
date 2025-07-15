# Spotify-Song-Recommender
“Clustering-based Spotify recommender system using PCA, K-Means, and similarity scoring.”
# 🎵 Spotify Song Recommender with Genre-Based Clustering

## 📌 Overview
This project presents a clustering-driven recommender system for Spotify songs, using audio features to uncover genre-based patterns and suggest similar tracks. It applies unsupervised machine learning to segment music by mood, energy, and rhythm — transforming listening data into intelligent playlist suggestions.

## 🧠 Key Methods & Technologies
- **Data Preprocessing**: Missing value removal, column standardization
- **Visualization**: Genre distribution, feature histograms, correlation matrix
- **Feature Scaling & PCA**: `StandardScaler` + `PCA` for 2D dimensionality reduction
- **Clustering**: `KMeans` with 5 clusters + visual cluster mapping
- **Recommendation Logic**: Suggests similar songs using cluster ID of selected track

## 💻 How to Run
```bash
git clone https://github.com/your-username/spotify-song-recommender.git
cd spotify-song-recommender
pip install -r requirements.txt
jupyter notebook
```
Open spotify_recommender.ipynb and run all cells to:
- Load & preprocess data
- Visualize cluster insights
- Generate song recommendations

🔍 Sample Usage
```bash
recommend_song(df, 'all the time - don diablo remix')
```
Output: 5 similar tracks from the same cluster

Included Files:
        File                                          Description 

 spotify_recommender.ipynb            = Full notebook with clustering, PCA, and recommender  |   data/clean_spotify.csv               = Preprocessed Spotify dataset  
 requirements.txt                     = Python packages used 
 Spotify Songs’ Genre Segmentation.md = Markdown export of full analysis 




⚖️ LISENCE 
This project is licensed under the MIT License.

