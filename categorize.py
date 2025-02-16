from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the preset categories with seed examples
preset_categories = {
    "environmental": [
        "The city is facing severe air pollution and waste management issues.",
        "Industrial emissions are causing environmental degradation.",
        "Deforestation is increasing due to urban expansion.",
        "Water pollution is affecting local rivers and drinking water supplies.",
        "Climate change is leading to more extreme weather events in the city."
    ],
    "housing": [
        "There is a shortage of affordable housing in the city.",
        "Housing prices have increased drastically over the past few years.",
        "Many residents are facing eviction due to rising rents.",
        "Homelessness has become a growing crisis.",
        "Substandard housing conditions pose health and safety risks."
    ],
    "infrastructure": [
        "Aging roads and bridges are in dire need of repair.",
        "Frequent power outages disrupt daily life and business operations.",
        "The city's drainage system is inadequate, leading to frequent flooding.",
        "Poorly maintained sidewalks make walking unsafe for pedestrians.",
        "Water supply shortages are affecting multiple neighborhoods."
    ],
    "crime/safety": [
        "Violent crime rates have increased significantly in the past year.",
        "There is a lack of police presence in high-crime areas.",
        "Street lighting is insufficient, making some areas dangerous at night.",
        "Vandalism and property crimes are rising in residential areas.",
        "Emergency response times are too slow due to understaffing."
    ],
    "healthcare": [
        "There is a shortage of doctors and healthcare facilities in the city.",
        "Many residents cannot afford necessary medical care.",
        "Emergency rooms are overcrowded and have long wait times.",
        "Access to mental health services is very limited.",
        "Public hospitals lack funding and face frequent supply shortages."
    ],
    "public spaces": [
        "There are not enough parks and recreational areas for residents.",
        "Existing public spaces are poorly maintained and unsafe.",
        "Green spaces are being lost due to urban development.",
        "There is a lack of seating and shaded areas in public places.",
        "Public restrooms are either unavailable or not well-maintained."
    ],
    "transportation": [
        "Public transportation is unreliable and overcrowded.",
        "Traffic congestion is worsening due to inadequate infrastructure.",
        "Bike lanes are poorly designed and not well-maintained.",
        "There is a lack of pedestrian-friendly areas and crosswalks.",
        "Public transit fares have become too expensive for many residents."
    ],
    "education": [
        "Public schools are underfunded and overcrowded.",
        "There is a shortage of qualified teachers in the district.",
        "School infrastructure is outdated and in poor condition.",
        "Access to quality education is unequal across neighborhoods.",
        "After-school programs and extracurricular activities are lacking."
    ],
    "economy/employment": [
        "Unemployment rates are high due to a lack of job opportunities.",
        "Small businesses are struggling to survive in the current economy.",
        "Wages have not kept up with the rising cost of living.",
        "Job training and skill development programs are insufficient.",
        "Many workers are facing job insecurity due to automation."
    ],
    "digital access/technology": [
        "Many areas in the city lack access to high-speed internet.",
        "Public Wi-Fi is limited and unreliable.",
        "The digital divide is preventing low-income residents from accessing online resources.",
        "Outdated technology in schools is hindering student learning.",
        "There are not enough public computer labs or digital literacy programs."
    ]
}

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute the embeddings for each seed text per category and calculate centroids
category_centroids = {}
for category, seed_texts in preset_categories.items():
    embeddings = model.encode(seed_texts)
    # Compute the centroid by averaging the embeddings
    centroid = np.mean(embeddings, axis=0)
    category_centroids[category] = centroid

def assign_category(text, model, category_centroids):
    # Encode the new text into an embedding
    text_embedding = model.encode([text])[0]  # [0] to get the vector from the list

    # Compute cosine similarity with each category centroid
    similarities = {}
    for category, centroid in category_centroids.items():
        # Compute the cosine similarity (using 2D arrays for compatibility)
        sim = cosine_similarity([text_embedding], [centroid])[0][0]
        similarities[category] = sim

    # Loop over the similarity scores and print each one
    print("Similarity Scores:")
    for category, score in similarities.items():
        print(f"  {category}: {score:.3f}")

    # Find the category with the highest similarity
    assigned_category = max(similarities, key=similarities.get)
    return assigned_category, similarities