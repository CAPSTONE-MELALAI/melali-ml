from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/recommendation")
async def create_recommendation(data: dict):
    try :
        idx_selected = data.get("idx_selected")
        budget = data.get("budget")
        days = data.get("days")
        lat_user = data.get("lat_user")
        long_user = data.get("long_user")
        is_accessibility = data.get("is_accessibility")
        
        if not idx_selected or not budget or not days or not lat_user or not long_user:
            raise HTTPException(status_code=400, detail="Bad Request")

        data_recom = create_recommendation(idx_selected, budget, days, lat_user, long_user, is_accessibility)
        return create_response(success=True, 
                               message="gg bang",
                               data=data_recom)
    except Exception as e:
        return create_response(success=False, 
                               message="nt bang",
                               data=[])

@app.post("/might-like/{id}")
async def create_10_top(id: int):
    try :
        if not id:
            raise HTTPException(status_code=400, detail="Bad Request")

        data_recom = might_like(id)
        return create_response(success=True, 
                               message="gg bang",
                               data=data_recom)
    except Exception as e:
        return create_response(success=False, 
                               message="nt bang",
                               data=[])
    
### HERE IS LOGIC
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

from fastapi.responses import JSONResponse

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_destination = pd.read_csv('../dataset/new_bali_dataset.csv', delimiter=';', header=0)
df_destination['price'].fillna(0, inplace=True)
df_destination['category'].fillna('unknown', inplace=True)


# Function for removing NonAscii characters
def _removeNonAscii(text):
    return "".join(i for i in text if  ord(i)<128)

# Function for converting into lower case
def make_lower_case(text):
    return text.lower()

# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Applying all the functions in description and storing as a cleaned_desc
df_destination['cleaned_desc'] = df_destination['description'].apply(_removeNonAscii)
df_destination['cleaned_desc'] = df_destination.cleaned_desc.apply(func = make_lower_case)
df_destination['cleaned_desc'] = df_destination.cleaned_desc.apply(func = remove_stop_words)
df_destination['cleaned_desc'] = df_destination.cleaned_desc.apply(func=remove_punctuation)
df_destination['cleaned_desc'] = df_destination.cleaned_desc.apply(func=remove_html)

def recommendation(index):
    
    place = df_destination.loc[index, 'place']
    category = df_destination.loc[index, 'category']
    # Matching the category with the dataset and reset the index
    data_category = df_destination[df_destination['category'] == category].reset_index(drop=True)
  
    # Convert the index into series
    indices = pd.Series(data_category.index, index=data_category['place'])
    
    # Converting the place description into vectors
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data_category['cleaned_desc'])
    
    # Calculating the similarity measures based on Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index corresponding to the place
    idx = indices[place]
    
    # Get the pairwise similarity scores 
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the places
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Scores of the 5 most similar places
    sim_scores = sim_scores[1:10]
    
    # Place indices
    place_indices = [i[0] for i in sim_scores]
   
    # Top 5 place recommendations
    records = data_category['place'].iloc[place_indices]
    return records

def sort_place(affordable_places,lat_user, long_user):
    # get latitude and longitude from affordable_places

    # calculate euclidean distance from user to places
    distances = np.sqrt(((affordable_places['lat'] - lat_user)**2) + ((affordable_places['long'] - long_user)**2))
    affordable_places['distance'] = distances
    
    # sort dataframe based on distance
    affordable_places = affordable_places.sort_values(by='distance', ascending=False)
    
    return affordable_places

def create_recommendation(idx_selected, budget, days, lat_user, long_user, is_accessibility=0):
    # create recommendation
    recommended_places = set()
    # selected_places = df_destination.iloc[idx_selected][['place', 'category']].values
    for idx in idx_selected:
        recommended_places.update(recommendation(idx))

    filtered_places = df_destination[df_destination['place'].isin(recommended_places)]
    
    # filter based on accessibility
    if is_accessibility == 1:
        filtered_places = filtered_places[filtered_places['is_accessibility'] == 1]
   
    #filter based on budget
    affordable_places = filtered_places[filtered_places['price'] <= budget / days]

    # Sort places by Google Maps Rating and then by Review Count
    affordable_places = affordable_places.sort_values(by=['rating', 'n_reviews'], ascending=[False, False])
    # return affordable_places
    # sort by distance
    affordable_places = sort_place(affordable_places, lat_user, long_user)

    # Create itinerary
    
    list_per_day = []
    places_per_day = 3
    all_places = filtered_places.sort_values(by=['rating', 'n_reviews'], ascending=[False, False])

    used_places = set()
    for day in range(1, days + 1):
        list_of_dest = []   
        daily_itinerary = affordable_places[~affordable_places['place'].isin(used_places)].head(places_per_day)
        
        # Fallback if no affordable places are left
        if daily_itinerary.empty:
            daily_itinerary = all_places[~all_places['place'].isin(used_places)].head(places_per_day)
        
        # Ensure at least one place per day
        while len(daily_itinerary) < places_per_day and not all_places[~all_places['place'].isin(used_places)].empty:
            additional_place = all_places[~all_places['place'].isin(used_places)].head(1)
            daily_itinerary = pd.concat([daily_itinerary, additional_place])
        
        for row in daily_itinerary.iterrows():
            dest_dict = {}
            dest_dict['place'] = row[1]['place']
            dest_dict['url'] = row[1]['url']
            dest_dict['address'] = row[1]['address']
            dest_dict['is_accessibility'] = row[1]['is_accessibility']
            dest_dict['rating'] = row[1]['rating']
            dest_dict['n_reviews'] = row[1]['n_reviews']
            dest_dict['price'] = row[1]['price']
            dest_dict['category'] = row[1]['category']
            dest_dict['description'] = row[1]['description']
            list_of_dest.append(dest_dict)
         
            
        list_per_day.append(list_of_dest)
 
        # Remove selected places from affordable_places and all_places to avoid duplicates
        used_places.update(daily_itinerary['place'])

    return list_per_day

### HERE IS LOGIC 2
df_user = pd.read_csv('../dataset/user_rating.csv')
place_num = len(df_user.Place_Id.unique())
n_users = len(df_user.User_Id.unique())

from sklearn.model_selection import train_test_split
train, test = train_test_split(df_user, test_size=0.2, random_state=42)

def create_model():
    # creating destination embedding path
    destination_input = Input(shape=[1])
    destination_embedding = Embedding(place_num+1, 5)(destination_input)
    destination_flat = Flatten()(destination_embedding)

    # creating user embedding path
    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users+1, 5)(user_input)
    user_flat = Flatten()(user_embedding)

    # concatenate features
    conc = Concatenate()([destination_flat, user_flat])

    # add fully-connected-layers
    x = Dense(128, activation='relu')(conc)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)

    # Create model and compile it
    model = Model([user_input, destination_input], output)
    model.compile('adam', 'mean_squared_error')
    history = model.fit([train.User_Id, train.Place_Id], train.Place_Ratings, epochs=20)
    return history, model

model = tf.keras.models.load_model('../model/model.h5')

def create_final_data(list_of_idx):
    new_df = df_destination[df_destination.index.isin(list_of_idx)].copy()
    list_of_dest = []

    for row in new_df.iterrows():
        dest_dict = {}
        dest_dict['place'] = row[1]['place']
        dest_dict['url'] = row[1]['url']
        dest_dict['address'] = row[1]['address']
        dest_dict['is_accessibility'] = row[1]['is_accessibility']
        dest_dict['rating'] = row[1]['rating']
        dest_dict['n_reviews'] = row[1]['n_reviews']
        dest_dict['price'] = row[1]['price']
        dest_dict['category'] = row[1]['category']
        dest_dict['description'] = row[1]['description']
        list_of_dest.append(dest_dict)
    
    return list_of_dest

def might_like(idx_user, model=model):
    tourism_data = np.array(list(set(df_user.Place_Id)))
    user = np.array([idx_user]*len(tourism_data))

    predictions = model.predict([user, tourism_data])
    predictions = np.array([a[0] for a in predictions])
    
    recommended_tourism_idx = (-predictions).argsort()[:10]
    
    #convert index to data place
    data = create_final_data(recommended_tourism_idx)
    return data

def create_response(success: bool, message:str, data):
    response = {
        "success": success,
        "message": message,
        "data": data
    }
    return JSONResponse(content=response)