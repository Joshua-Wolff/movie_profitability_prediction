import pandas as pd
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


####

films_credits = pd.read_json("data/raw/films_credits.json.zip")
films_infos = pd.read_json("data/raw/films_infos.json")

films_infos = films_infos.drop_duplicates(subset='id')
films_credits = films_credits.drop_duplicates(subset='id')

films_df = films_infos.merge(
    films_credits,
    how="left",
    on="id",  
)

del films_infos, films_credits

useful_columns = ['id', 'original_title', 'belongs_to_collection', 'release_date', 'runtime', 'budget', 'genres', \
     'original_language', 'spoken_languages',  'production_companies', 'production_countries', \
     'cast', 'crew', 'tagline', 'homepage', 'revenue']

films_df = films_df[useful_columns]

films_df = films_df[np.array(films_df['revenue'] > 0) & np.array(films_df['budget'] > 0)]

desired_rentability = 2

films_df["class"] = np.int8((films_df["revenue"] / films_df["budget"]) >= desired_rentability)

####

train_df, test_df = train_test_split(films_df, test_size=0.33, random_state=12)

mean_runtime = train_df['runtime'].mean()

train_df['runtime']=train_df['runtime'].fillna(0.0).replace(0.0, mean_runtime)
test_df['runtime']=test_df['runtime'].fillna(0.0).replace(0.0, mean_runtime)

####

train_df['release_date'] = pd.to_datetime(train_df['release_date'])
test_df['release_date'] = pd.to_datetime(test_df['release_date'])

def process_release_date(df):
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_weekday'] = df['release_date'].dt.weekday

process_release_date(train_df)
process_release_date(test_df)

####

train_df['collection_name'] = train_df['belongs_to_collection'].apply(lambda x: x['name'] if x != None else '')
test_df['collection_name'] = test_df['belongs_to_collection'].apply(lambda x: x['name'] if x != None else '')

train_df['has_collection'] = train_df['belongs_to_collection'].apply(lambda x: 1 if x != None else 0)
test_df['has_collection'] = test_df['belongs_to_collection'].apply(lambda x: 1 if x != None else 0)

train_df['has_homepage'] = train_df['homepage'].apply(lambda x: 0 if x == '' else 1)
test_df['has_homepage'] = test_df['homepage'].apply(lambda x: 0 if x == '' else 1)

train_df['has_a_tagline'] = train_df['tagline'].apply(lambda x: 0 if x == '' else 1)
test_df['has_a_tagline'] = test_df['tagline'].apply(lambda x: 0 if x == '' else 1)

train_df = train_df.drop(['belongs_to_collection', 'homepage', 'tagline'], axis=1)
test_df = test_df.drop(['belongs_to_collection', 'homepage', 'tagline'], axis=1)

####

list_of_languages = list(
    train_df['spoken_languages']
    .apply(lambda x: [i['english_name'] for i in x] if x != [] else [])
    .values
)

most_common_languages = Counter([language for list_ in list_of_languages for language in list_ ]).most_common(30)

train_df['num_languages'] = train_df['spoken_languages'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_languages'] = test_df['spoken_languages'].apply(lambda x: len(x) if x != [] else 0)

# We get all the english names of the spoken languages in the movie
train_df['all_languages'] = (
    train_df['spoken_languages']
    .apply(lambda x: [language['english_name'].lower() for language in x] if x != [] else [])
)

# Then for each one of the most common language, we look up if it is part 
# of the spoken language in the movie
top_30_languages = [language[0].lower() for language in most_common_languages]

for language in top_30_languages:
    train_df['language_' + language] = (
        train_df['all_languages']
        .apply(lambda x: 1 if language in x else 0)
    )

# We get all the english names of the spoken languages in the movie
test_df['all_languages'] = (
    test_df['spoken_languages']
    .apply(lambda x: [language['english_name'].lower() for language in x] if x != [] else [])
)

# Then for each one of the most common language, we look up if it is part 
# of the spoken language in the movie
for language in top_30_languages:
    test_df['language_' + language] = (
        test_df['all_languages']
        .apply(lambda x: 1 if language in x else 0)
    )

train_df.drop(columns=['spoken_languages', 'all_languages'], inplace=True)
test_df.drop(columns=['spoken_languages', 'all_languages'], inplace=True)


####


list_of_genres = list(
    train_df['genres']
    .apply(lambda x: [genre['name'] for genre in x] if x != [] else [])
    .values
)

most_common_genres = Counter([genre for list_ in list_of_genres for genre in list_ ]).most_common()

train_df['num_genres'] = train_df['genres'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_genres'] = test_df['genres'].apply(lambda x: len(x) if x != [] else 0)

# We get all the names of all the genres of the movie
train_df['all_genres'] = (
    train_df['genres']
    .apply(lambda x: [genre['name'].lower() for genre in x] if x != [] else [])
)

# Then for each one of the most common genre, we look up if it is part 
# of the genres of the movie
top_genres = [genre[0].lower() for genre in most_common_genres]

for genre in top_genres:
    train_df['genre_' + genre] = (
        train_df['all_genres']
        .apply(lambda x: 1 if genre in x else 0)
    )


# We get all the names of all the genres of the movie
test_df['all_genres'] = (
    test_df['genres']
    .apply(lambda x: [genre['name'].lower() for genre in x] if x != [] else [])
)

# Then for each one of the most common genre, we look up if it is part 
# of the genres of the movie

for genre in top_genres:
    test_df['genre_' + genre] = (
        test_df['all_genres']
        .apply(lambda x: 1 if genre in x else 0)
    )


train_df.drop(columns=['genres', 'all_genres'], inplace=True)
test_df.drop(columns=['genres', 'all_genres'], inplace=True)


####


list_of_companies = list(
    train_df['production_companies']
    .apply(lambda x: [company['name'] for company in x] if x != [] else [])
    .values
)

# We take the companies which have >= 20 occurences
most_common_companies = Counter([company for list_ in list_of_companies for company in list_ ]).most_common(27)

train_df['num_companies'] = train_df['production_companies'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_companies'] = test_df['production_companies'].apply(lambda x: len(x) if x != [] else 0)


# We get all the names of all the prod companies of the movie
train_df['all_companies'] = (
    train_df['production_companies']
    .apply(lambda x: [company['name'] for company in x] if x != [] else [])
)

# Then for each one of the 27 most common company, we look up if it is part 
# of the companies of the movie
top_companies = [company[0] for company in most_common_companies]

for company in top_companies:
    train_df['company_' + '_'.join(company.lower().split())] = (
        train_df['all_companies']
        .apply(lambda x: 1 if company in x else 0)
    )


# We get all the names of all the prod companies of the movie
test_df['all_companies'] = (
    test_df['production_companies']
    .apply(lambda x: [company['name'] for company in x] if x != [] else [])
)

# Then for each one of the 27 most common company, we look up if it is part 
# of the companies of the movie

for company in top_companies:
    test_df['company_' + '_'.join(company.lower().split())] = (
        test_df['all_companies']
        .apply(lambda x: 1 if company in x else 0)
    )


train_df.drop(columns=['production_companies', 'all_companies'], inplace=True)
test_df.drop(columns=['production_companies', 'all_companies'], inplace=True)


####


list_of_countries = list(
    train_df['production_countries']
    .apply(lambda x: [country['name'] for country in x] if x != [] else [])
    .values
)

# We take the countries which have >= 20 occurences
most_common_countries = Counter([country for list_ in list_of_countries for country in list_ ]).most_common(21)


train_df['num_countries'] = train_df['production_countries'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_countries'] = test_df['production_countries'].apply(lambda x: len(x) if x != [] else 0)


# We get all the names of all the prod countries of the movie
train_df['all_countries'] = (
    train_df['production_countries']
    .apply(lambda x: [country['name'] for country in x] if x != [] else [])
)

# Then for each one of the 21 most common country, we look up if it is part 
# of the countries of the movie
top_countries = [country[0] for country in most_common_countries]

for country in top_countries:
    train_df['country_' + '_'.join(country.lower().split())] = (
        train_df['all_countries']
        .apply(lambda x: 1 if country in x else 0)
    )

# We get all the names of all the prod countries of the movie
test_df['all_countries'] = (
    test_df['production_countries']
    .apply(lambda x: [country['name'] for country in x] if x != [] else [])
)

# Then for each one of the 21 most common country, we look up if it is part 
# of the countries of the movie

for country in top_countries:
    test_df['country_' + '_'.join(country.lower().split())] = (
        test_df['all_countries']
        .apply(lambda x: 1 if country in x else 0)
    )

train_df.drop(columns=['production_countries', 'all_countries'], inplace=True)
test_df.drop(columns=['production_countries', 'all_countries'], inplace=True)


#####


list_of_cast_names = list(
    train_df['cast']
    .apply(lambda x: [casted['name'] for casted in x] if x != [] else [])
    .values
)

# We take the countries which have >= 15 occurences
most_common_casted = Counter([casted for list_ in list_of_cast_names for casted in list_ ]).most_common(29)

train_df['num_casted'] = train_df['cast'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_casted'] = test_df['cast'].apply(lambda x: len(x) if x != [] else 0)


# We get all the names of all the casted persons the movie
train_df['all_casted'] = (
    train_df['cast']
    .apply(lambda x: [casted['name'] for casted in x] if x != [] else [])
)

# Then for each one of the 29 most common casted, we look up if it is part 
# of the casted persons in the movie
top_cast = [casted[0] for casted in most_common_casted]

for casted in top_cast:
    train_df['cast_' + '_'.join(casted.lower().split())] = (
        train_df['all_casted']
        .apply(lambda x: 1 if casted in x else 0)
    )


# We get all the names of all the casted persons the movie
test_df['all_casted'] = (
    test_df['cast']
    .apply(lambda x: [casted['name'] for casted in x] if x != [] else [])
)

# Then for each one of the 29 most common casted, we look up if it is part 
# of the casted persons in the movie
top_cast = [casted[0] for casted in most_common_casted]

for casted in top_cast:
    test_df['cast_' + '_'.join(casted.lower().split())] = (
        test_df['all_casted']
        .apply(lambda x: 1 if casted in x else 0)
    )

train_df.drop(columns=['cast', 'all_casted'], inplace=True)
test_df.drop(columns=['cast', 'all_casted'], inplace=True)


#####


list_of_crew_names = list(
    train_df['crew']
    .apply(lambda x: [crew_member['name'] for crew_member in x] if x != [] else [])
    .values
)

# We take the 30 most common crew members
most_common_crew_members = Counter([crew_member for list_ in list_of_crew_names for crew_member in list_ ]).most_common(30)

train_df['num_crew_members'] = train_df['crew'].apply(lambda x: len(x) if x != [] else 0)
test_df['num_crew_members'] = test_df['crew'].apply(lambda x: len(x) if x != [] else 0)

# We get the names of all the crew members for the movie
train_df['all_crew_members'] = (
    train_df['crew']
    .apply(lambda x: [crew_member['name'] for crew_member in x] if x != [] else [])
)

# Then for each one of the 30 most common crew members, we look up if it is part 
# of the crew members for the movie
top_crew_members = [crew_member[0] for crew_member in most_common_crew_members]

for crew_member in top_crew_members:
    train_df['crew_' + '_'.join(crew_member.lower().split())] = (
        train_df['all_crew_members']
        .apply(lambda x: 1 if crew_member in x else 0)
    )


# We get the names of all the crew members for the movie
test_df['all_crew_members'] = (
    test_df['crew']
    .apply(lambda x: [crew_member['name'] for crew_member in x] if x != [] else [])
)

# Then for each one of the 30 most common crew members, we look up if it is part 
# of the crew members for the movie

for crew_member in top_crew_members:
    test_df['crew_' + '_'.join(crew_member.lower().split())] = (
        test_df['all_crew_members']
        .apply(lambda x: 1 if crew_member in x else 0)
    )

train_df.drop(columns=['crew', 'all_crew_members'], inplace=True)
test_df.drop(columns=['crew', 'all_crew_members'], inplace=True)


###################################@


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

#y_train = train_df['class'].copy()
#y_test = test_df['class'].copy()

info_columns = ['id', 'original_title', 'release_date']


train_df.drop(columns=info_columns, inplace=True)
test_df.drop(columns=info_columns, inplace=True)


train_df['collection_name'] = train_df['collection_name'].apply(lambda x: '_'.join(x.lower().split()))
test_df['collection_name'] = test_df['collection_name'].apply(lambda x: '_'.join(x.lower().split()))


categorical_cols =['original_language', 'collection_name']

one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoder.fit(train_df[categorical_cols])

encoded_cols = list(one_hot_encoder.get_feature_names(categorical_cols))


train_df[encoded_cols] = one_hot_encoder.transform(train_df[categorical_cols])
test_df[encoded_cols] = one_hot_encoder.transform(test_df[categorical_cols])

train_df.drop(columns=categorical_cols, inplace=True)
test_df.drop(columns=categorical_cols, inplace=True)


#########################

train_df.to_csv('train.csv')
train_df.to_csv('public/train.csv')

test_df.to_csv('test.csv')
test_df.to_csv('public/test.csv')



