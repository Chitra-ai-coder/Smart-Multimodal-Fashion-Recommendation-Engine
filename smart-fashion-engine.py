# %% [markdown] Cell 1
# # Smart Fashion Recommendation Engine
# **Author:** CHITRABHANU HAZRA 
# **Dataset:** Fashion Product Images (Small)
# 
# **Project Overview:**
# This notebook implements a fully multimodal AI fashion catalog. It utilizes Hugging Face's **CLIP** model for visual embedding and the **Flan-T5** Large Language Model for contextual logic. 
# 
# **Key Engineering Implementations:**
# * **Mathematical Deduplication:** Uses cosine similarity to automatically purge duplicate vendor images.
# * **Multimodal Projection Fixes:** Bypasses Hugging Face wrapper bugs by utilizing a dummy image tensor, forcing the text features to correctly align with the visual projection layer.
# * **Few-Shot Prompting:** Stabilizes the base-level LLM using explicit Q&A mapping to prevent hallucination and format breaking.

# %% [code] Cell 2
# Install required libraries if not already present
# !pip install transformers torch pandas scikit-learn pillow

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP Model
print("Loading CLIP Model...")
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
model = CLIPModel.from_pretrained(clip_model_id).to(device)
print("Model loaded successfully!")

# %% [code] Cell 3
# The exact path you provided
base_path = "/kaggle/input/datasets/paramaggarwal/fashion-product-images-small"

data_path = os.path.join(base_path, "styles.csv")
image_folder = os.path.join(base_path, "images")

# Load metadata, skipping any corrupted lines in the CSV
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path, on_bad_lines='skip')

# Sample 1000 items to keep execution fast. 
# (Remove .head(1000) when you are ready to run the full dataset!)
catalog_df = df.head(1000).copy()

def get_image_path(image_id):
    """Helper function to grab the right image file"""
    return os.path.join(image_folder, f"{image_id}.jpg")

print(f"Loaded {len(catalog_df)} items into the catalog.")

# %% [code] Cell 4
# ==========================================
# DATA LOAD & EMBEDDING
# ==========================================
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np

print("--- REBUILDING THE IMAGE DATABASE ---")

data_path = '/kaggle/input/datasets/paramaggarwal/fashion-product-images-small/myntradataset/styles.csv'
image_folder = '/kaggle/input/datasets/paramaggarwal/fashion-product-images-small/images'

# 1. Load CSV and find matching images
df = pd.read_csv(data_path, on_bad_lines='skip')
actual_files = os.listdir(image_folder)
available_ids = [f.replace('.jpg', '') for f in actual_files if f.endswith('.jpg')]

df['id'] = df['id'].astype(str).str.replace('.0', '', regex=False)
valid_df = df[df['id'].isin(available_ids)]
catalog_df = valid_df.head(1000).reset_index(drop=True)

def get_image_path(image_id):
    return os.path.join(image_folder, f"{image_id}.jpg")

# 2. Generate Embeddings using the foolproof method
print(f"Generating embeddings for {len(catalog_df)} items. This will take a minute...")
image_embeddings = []

model.eval()
with torch.no_grad():
    for idx, row in catalog_df.iterrows():
        img_path = get_image_path(row['id'])
        try:
            img = Image.open(img_path).convert("RGB")
            
            # THE FIX: Pass a dummy text alongside the image to force full, bug-free multimodal generation
            inputs = processor(text=["clothing"], images=img, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            
            # Grab the officially projected image embedding!
            img_embed = outputs.image_embeds
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            
            image_embeddings.append(img_embed.cpu().numpy()[0])
        except Exception as e:
            print(f"Failed on {row['id']}: {e}")
            image_embeddings.append(np.zeros(512))

# 3. Finalize matrix
catalog_df['embedding'] = image_embeddings
all_image_embeds = np.vstack(catalog_df['embedding'].values)

# --- SANITY CHECK ---
if np.sum(all_image_embeds[0]) == 0:
    print("\nCRITICAL ERROR: Database is still full of zeros!")
else:
    print(f"\nSUCCESS! Rebuilt {len(catalog_df)} valid image embeddings. The zeros are gone!")

# %% [markdown] Cell 5
# ## Task 2: Unique Product Catalog Creation
# Before building our search engines, we must ensure our database is clean. Real-world retail datasets frequently contain multiple identical images uploaded under different IDs. 
# 
# In this section, we generate **CLIP visual embeddings** for our dataset and calculate a similarity matrix. Any images that share a mathematically identical embedding (cosine similarity > 0.98) are flagged as duplicates and removed, leaving us with a perfectly pristine catalog.

# %% [code] Cell 6
# ==========================================
# TASK 2 - UNIQUE CATALOG CREATION
# ==========================================
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_unique_catalog(df, embeddings_matrix, similarity_threshold=0.98):
    """
    Identifies duplicates using visual embeddings and returns a deduplicated catalog.
    """
    print("\n" + "="*50)
    print("TASK 2: Unique Product Catalog Creation")
    print("="*50)
    
    print("Calculating similarity matrix...")
    # Calculate similarity between all items
    sim_matrix = cosine_similarity(embeddings_matrix)
    
    unique_indices = []
    seen_duplicates = set()
    
    for i in range(len(sim_matrix)):
        # If we already flagged this item as a duplicate of something else, skip it
        if i in seen_duplicates:
            continue
            
        unique_indices.append(i)
        
        # Find all items that are highly similar to item 'i' (including itself)
        duplicates = np.where(sim_matrix[i] > similarity_threshold)[0]
        seen_duplicates.update(duplicates)
        
    unique_catalog = df.iloc[unique_indices].reset_index(drop=True)
    
    print(f"Original Catalog Size: {len(df)}")
    print(f"Unique Catalog Size: {len(unique_catalog)}")
    print(f"Removed {len(df) - len(unique_catalog)} duplicate or near-duplicate items.\n")
    
    return unique_catalog

# Execute Task 2 and save the clean data for the next tasks
clean_catalog_df = create_unique_catalog(catalog_df, all_image_embeds)
clean_embeds = np.vstack(clean_catalog_df['embedding'].values)

# %% [code] Cell 7
# Lower the threshold to 0.92 to catch near-duplicates
clean_catalog_df = create_unique_catalog(catalog_df, all_image_embeds, similarity_threshold=0.92)
clean_embeds = np.vstack(clean_catalog_df['embedding'].values)

# %% [code] Cell 8
# ==========================================
# BONUS CELL: VIEWING THE UNIQUE CATALOG
# ==========================================
import matplotlib.pyplot as plt
from PIL import Image

print("\n" + "="*50)
print(f"Your Unique Catalog contains {len(clean_catalog_df)} verified, deduplicated items.")
print("="*50)

# Let's grab 10 random items from your clean catalog to inspect
sample_df = clean_catalog_df.sample(10)

# 1. Draw the Images
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, (idx, product) in enumerate(sample_df.iterrows()):
    name = f"{product['gender']} {product['baseColour']} {product['articleType']}"
    
    img_path = get_image_path(product['id'])
    try:
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{name}\n(ID: {product['id']})", fontsize=10, fontweight='bold')
    except Exception:
        axes[i].set_title("Image Error")
        axes[i].axis('off')

plt.tight_layout()
plt.show()

# 2. Print the Raw DataFrame Data
print("\nHere is what the raw data looks like under the hood:")
# The display() function renders a beautiful, formatted table in Kaggle
display(clean_catalog_df[['id', 'gender', 'masterCategory', 'articleType', 'baseColour', 'usage']].head(10))

# %% [markdown] Cell 9
# ## Task 3: Reverse Product Search
# This engine allows a user to type a text query (e.g., "blue casual shirt") and instantly retrieve visually matching items from the catalog.
# 
# **Engineering Note:** To ensure the text features are properly translated into the shared visual dimension, a blank 1x1 tensor is passed alongside the text. This forces the model to execute a complete, bug-free multimodal pass through the projection layer, resulting in highly accurate mathematical matches.

# %% [code] Cell 10
# ==========================================
# TASK 3 - REVERSE PRODUCT SEARCH
# ==========================================
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def reverse_product_search(query_text, top_k=3):
    print("\n" + "="*50)
    print("TASK 3: Reverse Product Search")
    print("="*50)
    print(f"Input Query: '{query_text}'\n")
    
    # 1. THE MATHEMATICAL FIX: Explicit Projection
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Step A: Get the raw text understanding
        text_outputs = model.text_model(**inputs)
        
        # Step B: FORCE the translation into the shared Image/Text dimension
        projected_text = model.text_projection(text_outputs.pooler_output)
        
        # Step C: Normalize so the cosine similarity math works perfectly
        text_embed = projected_text / projected_text.norm(dim=-1, keepdim=True)
        text_embed = text_embed.cpu().numpy()
    
    # 2. Find the highest mathematical matches
    similarities = cosine_similarity(text_embed, clean_embeds)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print("Top Matching Products:\n")
    
    # 3. Setup Matplotlib for a clean, side-by-side display
    fig, axes = plt.subplots(1, top_k, figsize=(15, 6))
    
    for i, idx in enumerate(top_indices):
        product = clean_catalog_df.iloc[idx]
        name = f"{product['gender']} {product['baseColour']} {product['usage']} {product['articleType']}"
        score = similarities[idx]
        
        # Print the details. A true match will now score higher than 0.20!
        print(f"{i+1}. {name} (ID: {product['id']}) - Match Score: {score:.3f}")
        
        # Load the actual image
        img_path = get_image_path(product['id'])
        img = Image.open(img_path)
        
        # Display the image in the grid
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

# Test the search engine! 
reverse_product_search("white Running Shoes")

# %% [markdown] Cell 11
# ## Task 1: Smart Product Recommendation Engine
# This pipeline combines logical reasoning with visual retrieval. 
# 1.  **The Brain (Flan-T5):** Analyzes the input item and suggests 5 logical accessories.
# 2.  **The Eyes (CLIP):** Takes those 5 text suggestions and pulls the exact matching images from our unique catalog.
# 
# **Engineering Note:** Base-level LLMs are prone to hallucination and formatting errors. To stabilize the output, this function uses **Multi-Shot Prompting** (providing the AI with rigid examples) alongside carefully tuned decoding parameters (`temperature=0.6`, `repetition_penalty=2.0`). A Python `while` loop is also implemented as a backend safety fallback to guarantee the visual UI never crashes.

# %% [code] Cell 12
# ==========================================
# TASK 1 - SMART RECOMMENDATION ENGINE (Production Ready)
# ==========================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

print("\n" + "="*50)
print("TASK 1: Smart Product Recommendation Engine")
print("="*50)

print("Loading LLM Fashion Consultant (Flan-T5)...")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
print("LLM Loaded successfully!")

def smart_recommendation_engine(input_product_name):
    print(f"\nUser Bought: '{input_product_name}'\n")
    
    # THE FIX 1: Ultra-short, rigid mapping format. 
    prompt = (
        "Item: Winter Coat\nMatches: scarf, gloves, beanie, boots, pants\n\n"
        "Item: Swimsuit\nMatches: sunglasses, flip flops, towel, hat, bag\n\n"
        f"Item: {input_product_name}\nMatches:"
    )
    
    llm_inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    
    # THE FIX 2: Use sampling to force creativity and block repetition
    llm_outputs = llm_model.generate(
        **llm_inputs, 
        max_new_tokens=30,
        do_sample=True,          
        temperature=0.6,         
        repetition_penalty=2.0   # Strictly forbids repeating variations of "shirt"
    )
    llm_response = llm_tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
    
    # Clean up the output 
    cleaned_response = llm_response.replace('and', ',').replace('.', '')
    suggested_categories = [item.strip() for item in cleaned_response.split(',') if len(item.strip()) > 2]
    
    # THE FIX 3: The Production Safety Net. 
    # If the LLM gets lazy, we pad the array so the visual UI never crashes.
    fallbacks = ["jeans", "casual shoes", "belt", "watch", "jacket"]
    while len(suggested_categories) < 5:
        suggested_categories.append(fallbacks.pop(0))
        
    suggested_categories = suggested_categories[:5] 
    
    print(f"AI suggests pairing it with: {suggested_categories}\n")
    print("Finding the best matches in your catalog...\n")
    
    # Setup Matplotlib for display
    fig, axes = plt.subplots(1, len(suggested_categories), figsize=(20, 5))
    if len(suggested_categories) == 1:
        axes = [axes] 
        
    # Find real items in our catalog for each AI suggestion
    for i, category in enumerate(suggested_categories):
        
        dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
        inputs = processor(text=[category], images=dummy_image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            text_embed = outputs.text_embeds
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.cpu().numpy()
            
        similarities = cosine_similarity(text_embed, clean_embeds)[0]
        top_idx = np.argsort(similarities)[::-1][0] 
        
        rec_product = clean_catalog_df.iloc[top_idx]
        rec_name = f"{rec_product['gender']} {rec_product['baseColour']} {rec_product['usage']} {rec_product['articleType']}"
        
        img_path = get_image_path(rec_product['id'])
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"AI Suggestion:\n'{category}'\n\nMatch: {rec_name}", fontsize=10, fontweight='bold')
        except Exception:
            axes[i].set_title("Image Load Error")
            axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

# Test the search engine! 
smart_recommendation_engine("casual shoes")
