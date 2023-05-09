# to generate the embedding based on current distance

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.manifold import MDS

def embed_dist(distance_mtrx, embed_dim):
    n = distance_mtrx.shape[0]
    # embeddings = np.random.rand(n, embed_dim)
    #
    # iter = 0
    # resulting_distances = np.zeros_like(distance_mtrx)
    # while iter < 100 and (resulting_distances!=distance_mtrx).any():
    #
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             # Compute the current distance between embeddings i and j
    #             current_distance = np.linalg.norm(embeddings[i] - embeddings[j])
    #
    #             # Compute the scaling factor
    #             scaling_factor = distance_mtrx[i, j] / current_distance
    #
    #             # Update the embedding j by adjusting the distance from embedding i
    #             embeddings[j] = embeddings[i] + scaling_factor * (embeddings[j] - embeddings[i])
    #
    #     resulting_distances = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(n):
    #             resulting_distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    #     iter += 1

    embedding = MDS(n_components=embed_dim, dissimilarity='precomputed')
    embeddings = embedding.fit_transform(distance_mtrx)
    resulting_distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            resulting_distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

    print("Resulting pairwise Euclidean distances:")
    print(resulting_distances)

    print('Target pairwise Euclidean distances:')
    print(distance_mtrx)

    return embeddings, resulting_distances




# generate distance matrix
data_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data'
mote_loc = pd.read_csv(data_path + '/labapp3-positions.txt', sep='\s+', names=['moteid', 'x', 'y'])
moteids_to_remove = [5, 15, 18, 28, 54, 55, 56, 58]
filtered_df = mote_loc[~mote_loc['moteid'].isin(moteids_to_remove)]

xy_loc = filtered_df[['x', 'y']].values
distances = pdist(xy_loc)
square_distances = squareform(distances)

embed_dim = 64
embeddings, resulting_distances = embed_dist(square_distances, embed_dim)
np.save(data_path + '/embedding', embeddings)


#%% plot the locations
import matplotlib.pyplot as plt

plt.scatter(filtered_df['x'], filtered_df['y'])
for index, row in filtered_df.iterrows():
    plt.annotate(row['moteid'], (row['x'], row['y']), textcoords="offset points", xytext=(0,5), ha='center')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Mote location')
plt.grid(True)
plt.show()