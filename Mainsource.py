import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = 'D/Guvi/Train.csv'
spotify_data = pd.read_csv(dataset_path)

print(spotify_data.info())

print(spotify_data.head())

print(spotify_data.describe())

plt.figure(figsize=(10, 6))
sns.histplot(spotify_data['popularity'], bins=30, kde=True)
plt.title('Distribution of Popularity Scores')
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.show()

correlation_matrix = spotify_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Audio Features')
plt.show()
