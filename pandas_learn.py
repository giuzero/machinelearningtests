import pandas as pd

fruits = pd.DataFrame([{'Apples':30,'Bananas':21}])
fruit_sales = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index=['2017 Sales','2018 Sales'])

ingredients = pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'], name="Dinner")
#same as
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
recipe = pd.Series(quantities, index=items, name='Dinner')

#read csv
reviews = pd.read_csv("./pandas_learn_files/winemag-data-130k-v2.csv", index_col=0)

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv("./pandas_learn_files/animals.csv")

#Select the description column from reviews and assign the result to the variable desc.
desc = reviews.description

#Select the first value from the description column of reviews, assigning it to variable first_description.
first_description = reviews["description"][0]
#same as
first_description = reviews.description.iloc[0]
#same as
first_description = reviews.description.loc[0]
#same as 
first_description = reviews.description[0]

#Select the first row of data (the first record) from reviews, assigning it to the variable first_row.
first_row = reviews.iloc[0]

#Select the first 10 values from the description column in reviews, assigning the result to variable first_descriptions
first_descriptions = reviews.loc[:9,"description"]
#same as
first_descriptions = desc.head(10)
#same as
first_descriptions = reviews.loc[:9, "description"]
#same as
first_descriptions = reviews.description.iloc[:10]

#Select the records with index labels 1, 2, 3, 5, and 8, assigning the result to the variable sample_reviews.
sample_reviews = reviews.iloc[[1,2,3,5,8],:]
#same as
indices = [1, 2, 3, 5, 8]
sample_reviews = reviews.loc[indices]

#Create a variable df containing the 
# country, province, region_1, and region_2 columns 
# of the records with the index labels 0, 1, 10, and 100.
index_list=[0,1,10,100]
cols=["country", "province", "region_1", "region_2"]
df = reviews.loc[index_list,cols]

#Create a DataFrame italian_wines containing reviews of wines made in Italy
italian_wines = reviews.loc[reviews.country=='Italy']

#Create a DataFrame top_oceania_wines containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.
top_oceania_wines = reviews.loc[(reviews.points>=95) & ((reviews.country=="Australia") | (reviews.country=="New Zealand"))]
#same as
top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Australia', 'New Zealand']))
    & (reviews.points >= 95)
]

#Create a variable df containing the country and variety columns of the first 100 records.
cols = ['country', 'variety']
df = reviews.loc[:99, cols]
#same as
cols_idx = [0, 11]
df = reviews.iloc[:100, cols_idx]

#What is the median of the points column in the reviews DataFrame?
median_points = reviews.points.median()
#distinct countries in the dataset
countries = reviews.country.unique()
#How often does each country appear in the dataset?
reviews_per_country = reviews.country.value_counts()

#Create variable centered_price containing a version of the price column with the mean price subtracted.
#(Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.)
rpm = reviews.price.mean()
centered_price = reviews.price.map(lambda p: p - rpm)

#Which wine is the "best bargain"?
#Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

#Is a wine more likely to be "tropical" or "fruity"?
#  Create a Series descriptor_counts counting how many times each of these two words appears in the description column in the dataset.
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

#We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand 
# - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 
# but less than 95 is 2 stars. Any other score is 1 star.
#Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
#Create a series star_ratings with the number of stars corresponding to each review in the dataset.

def star_assignment(row):
    if row.country == 'Canada' or row.points >= 95:
        return 3
    elif row.points >=85 and row.points<95:
        return 2
    else:
        return 1
star_ratings = reviews.apply(star_assignment, axis='columns')