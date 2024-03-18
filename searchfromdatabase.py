import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
#streamlit rekomendacija webui

def search_articles():
    # Get the keyword
    keyword = entry.get()
    # Search for a keyword or phrase
    keyword_vector = vectorizer.transform([keyword])

    # Calculate cosine similarity between query vector and document vectors
    similarities = cosine_similarity(keyword_vector, X)

    # Threshold for cosine similarity
    threshold = 0.1

    # Find the most similar document indices above the threshold
    relevant_indices = [idx for idx, sim in enumerate(similarities.flatten()) if sim > threshold]

    # Sort relevant indices by cosine similarity score (highest first)
    relevant_indices = sorted(relevant_indices, key=lambda idx: similarities[0, idx], reverse=True)

    # Display search results
    if not relevant_indices:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "No search result for this phrase/keyword :-(")
        result_text.config(state=tk.DISABLED)
    else:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        i = 1
        for idx in relevant_indices:
            result_text.insert(tk.END, f"{i}. {df['Headline'][idx]}\n\n")
            i = i + 1
            #result_text.insert(tk.END, f"Similarity Score: {similarities[0, idx]}\n\n")
        result_text.config(state=tk.DISABLED)

    

def back_button_clicked():
    entry.delete(0, tk.END)
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.config(state=tk.DISABLED)

# Function to display articles for selected cluster group along with top 5 keywords
def display_cluster_articles(event):
    selected_group = cluster_combobox.get()
    selected_cluster_idx = int(selected_group.split()[1]) - 1
    
    # Find cluster indices
    cluster_indices = np.where(kmeans.labels_ == selected_cluster_idx)[0]
    
    # Display top 5 keywords for the selected cluster
    clusters_text.config(state=tk.NORMAL)
    clusters_text.delete(1.0, tk.END)
    clusters_text.insert(tk.END, "Top 5 Keywords:\n")
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    top_keywords = [terms[ind] for ind in order_centroids[selected_cluster_idx, :5]]
    clusters_text.insert(tk.END, ', '.join(top_keywords) + "\n\n")
    
    # Display articles for the selected cluster
    clusters_text.insert(tk.END, "Articles:\n")
    for idx in cluster_indices:
        clusters_text.insert(tk.END, f"{idx} - {df['Headline'][idx]}\n")
    
    clusters_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Article Search")
root.geometry("800x600")
root.resizable(True, True)

# Search Frame
search_frame = tk.Frame(root)
search_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

label = tk.Label(search_frame, text="Enter keyword/phrase:")
label.pack()

entry = tk.Entry(search_frame)
entry.pack()

search_button = tk.Button(search_frame, text="Search", command=search_articles)
search_button.pack()

back_button = tk.Button(search_frame, text="Delete", command=back_button_clicked)
back_button.pack()

result_text = tk.Text(search_frame, wrap=tk.WORD, height=20, width=50)
result_text.pack(fill=tk.BOTH, expand=True)
result_text.config(state=tk.DISABLED)

# Clusters Frame
clusters_frame = tk.Frame(root)
clusters_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

clusters_label = tk.Label(clusters_frame, text="Clusters")
clusters_label.pack()

# Dropdown menu for cluster groups
cluster_combobox = ttk.Combobox(clusters_frame, state="readonly")
cluster_combobox.pack()
cluster_combobox.bind("<<ComboboxSelected>>", display_cluster_articles)


clusters_text = tk.Text(clusters_frame, wrap=tk.WORD, height=20, width=50)
clusters_text.pack(fill=tk.BOTH, expand=True)
clusters_text.config(state=tk.DISABLED)

    # Read the .csv file
df = pd.read_csv('CNN_Articels_clean.csv')

    # Extract the specified column
preprocessed_text = df['Article text']

    # TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)
X = vectorizer.fit_transform(preprocessed_text)

    # K-means clustering
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Populate the dropdown menu with cluster groups
cluster_combobox["values"] = [f"Group {i+1}" for i in range(k)]


root.mainloop()
