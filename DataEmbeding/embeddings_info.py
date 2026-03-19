# Embeddings

# * Embeddings are noting but vector representation of text
# * To store these vector we need vector database
# * If we query a vector db, it won't find exact word, instead it will find similar words
# * Cosine Similarity, metric used to find out similarity between two vector representation
# * Use embedding models to embed test

# Why Embeddings, why do we need it at first place ?

# Feature Representation Example

# | Marvel | Action | Comedy | Suspense |
# |--------|--------|--------|----------|
# | Iron Man | 0.95 | 0.2 | 0.6 |
# | Hulk | 0.96 | 0.4 | 0.7 |
# | Sherlock Holmes | 0.6 | 0.85 | 0.9 |

# Example embeddings

# Let's say I am watching iron man, it will recommend hulk because cosine similaity between hulk and iron man is almost same when plotted on an xyz graph

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

animal_embeddings = {
    "dog": [0.95, 0.2],
    "cat": [0.96, 0.6],
    "bird": [0.6, 0.85],
    "car": [-0.5, 0.2],
    "truck": [-0.7, 0.3],
    "elephant": [0.9, 0.3],
    "horse": [0.85, 0.4],
    "fish": [0.7, 0.5],
    "snake": [0.65, 0.55],
    "chair": [0.5, 0.7],
    
}


fig,ax = plt.subplots(figsize=(8,10))

for i,word in enumerate(animal_embeddings):
    ax.scatter(animal_embeddings[word][0],animal_embeddings[word][1], s=100)
    ax.annotate(word, (animal_embeddings[word][0], animal_embeddings[word][1]), textcoords="offset points", xytext=(0,10), ha='center')

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
plt.title('2D Visualization of Animal Embeddings')
plt.show()

# Measuring Cosine Similarity

# - Results are close to 1: very similar
# - Results are close to 0: No related
# - Results are close to -1: Opposite

word_embeddings = {
    "ironman": [0.95, 0.2, 0.6],
    "The Dictator": [-0.2, 0.9, 0.1],
    "hulk": [0.96, 0.4, 0.7],
    "sherlockholmes": [0.6, 0.85, 0.9],
    "batman": [0.5, 0.7, 0.8],
    "The Dark Knight": [0.8, 0.1, 0.7],
    "The Avengers": [0.9, 0.3, 0.5],
    "The Dark Knight Rises": [0.9, 0.1, 0.8],
    
}



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for i,word in enumerate(word_embeddings):
    ax.scatter(word_embeddings[word][0],word_embeddings[word][1],word_embeddings[word][2], s=100)
    ax.text(word_embeddings[word][0],word_embeddings[word][1],word_embeddings[word][2],s=word,fontsize=12)

ax.set_xlabel('Action')
ax.set_ylabel('Comedy')
ax.set_zlabel('Suspense')
plt.title('3D Visualization of Movie Embeddings')
plt.show()

import numpy as np

def cosine_similarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    similarity = numerator / denominator
    
    if similarity > 0.9:
        return similarity, "Very similar"
    elif similarity > 0.5:
        return similarity, "Similar"
    elif similarity > 0.1:
        return similarity, "Not similar"
    else:
        return similarity, "Opposite"

print(f"Cosine Similarity between ironman and hulk: {cosine_similarity(word_embeddings['ironman'], word_embeddings['hulk'])}")
print(f"Cosine Similarity between ironman and The Dictator: {cosine_similarity(word_embeddings['ironman'], word_embeddings['The Dictator'])}")
print(f"Cosine Similarity between cat and truck: {cosine_similarity(animal_embeddings['cat'], animal_embeddings['truck'])}")

# Embedding Models
