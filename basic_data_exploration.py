import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def chap22():
    print("2 - Intro to ML - Basic Data Exploration")
    melbourne_file_path = 'melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path) 
    print(melbourne_data.describe()) # summary
    print(melbourne_data)


def chap23():
    print("3 - Intro to ML - Your First Machine Learning Model")
    melbourne_file_path = 'melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    # dropna drops missing values (think of na as "not available")
    melbourne_data = melbourne_data.dropna(axis=0)
    print(melbourne_data.columns)
    print(type(melbourne_data.columns))
    print(melbourne_data.Price)

    #prediction target
    y=melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 
    'Lattitude','Longtitude']
    X = melbourne_data[melbourne_features]
    #summary
    print(X.describe())
    #first part
    print(X.head())

    #Decision tree with sklearn [from sklearn.tree import DecisionTreeRegressor]
    melbourne_model=DecisionTreeRegressor(random_state=1)
    # Fit model
    melbourne_model.fit(X, y)
    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))

def chap24():
    print("4 - Intro to ML - Model Validation")
    print("on top: 'from sklearn.metrics import mean_absolute_error'")
    melbourne_file_path = 'melb_data.csv' 
    melbourne_data = pd.read_csv(melbourne_file_path) #load
    melbourne_data = melbourne_data.dropna(axis=0) #drop not available data
    y=melbourne_data.Price #set target
    X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 
    'Lattitude','Longtitude']] #set features
    melbourne_model=DecisionTreeRegressor(random_state=0) #define model
    melbourne_model.fit(X, y) #fit
    predicted_home_prices = melbourne_model.predict(X) #predict
    #Let's do an error: let's compute "In-Sample" scores intead of MAE really
    print("In-score problem: using whole data to build and same whole dataset to validate\nFAKE MAE:{}".format(mean_absolute_error(y, predicted_home_prices))) #mae on the same model of the prediction
    print("\n\nTRAIN_TEST_SPLIT\non top: 'from sklearn.model_selection import train_test_split'")
    #prediction could seem accurate but it could be so wrong
    #rebuild model: 
    # split design target and validation target, 
    # split design features and validation features
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)
    val_predictions = melbourne_model.predict(val_X)
    print("One part of dataset to design, the other to validate\nREAL MAE:{}".format(mean_absolute_error(val_y, val_predictions)))
    #this model sucks more than you thought...

def chap25():
    print("5 - Intro to ML - Underfitting and Overfitting")
    #Overfitting: model matches the training data almost perfectly
    #           poor in validations with new data
    #           Example: very deep tree
    #Underfitting: predoctions far off by most training and validation data
    #           pattern not found
    #           Example: shallow tree
    
    #SETUP
    melbourne_file_path = 'melb_data.csv' 
    melbourne_data = pd.read_csv(melbourne_file_path) #load
    melbourne_data = melbourne_data.dropna(axis=0) #drop not available data
    y=melbourne_data.Price #set target
    X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 
    'Lattitude','Longtitude']] #set features
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # I NEED A SWEET SPOT HEIGHT!!!
    
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return(mae)
    maes = {}
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: {}  \t\t Mean Absolute Error:  {}".format(max_leaf_nodes, my_mae))
        maes.update({max_leaf_nodes:my_mae})
    best_leaf_number = min(maes, key=maes.get)
    print("The model works best with {} leafs, with lower mae {}".format(best_leaf_number, maes[best_leaf_number]))

    """
    REWRITE:

    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    maes = { leafs_number : get_mae(leafs_number, train_X, val_X, train_y, val_y) for leafs_number in candidate_max_leaf_nodes}

    # Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
    best_tree_size = min(maes, key=maes.get)
    """

def chap26():
    print("6 - Intro to ML - Random Forest")
    #Do I want a deep decision tree or a shallow tree? Tough question; just use a RandomForest to test aa lot of trees
    #SETUP
    melbourne_file_path = 'melb_data.csv' 
    melbourne_data = pd.read_csv(melbourne_file_path) #load
    melbourne_data = melbourne_data.dropna(axis=0) #drop not available data
    y=melbourne_data.Price #set target
    X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 
    'Lattitude','Longtitude']] #set features
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # Let's try a random forest!!!
    print("on top: from sklearn.ensemble import RandomForestRegressor")
    forest_model = RandomForestRegressor(random_state=0)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print("Random forest predicion error: {}".format(mean_absolute_error(val_y, melb_preds)))

    #I want to compare it with the best tree model
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return(mae)

    maes = { leafs_number : get_mae(leafs_number, train_X, val_X, train_y, val_y) for leafs_number in range(5,5000)}
    print("Best prediction error with a tree I got is {} with {} leafs".format(maes[min(maes, key=maes.get)], min(maes, key=maes.get)))


def chap31():
    print("1 - Intermediate ML - Intorduction")
    X_full = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./kaggle_house_price_competition/test.csv', index_col='Id')

    # Obtain target and predictors
    y = X_full.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = X_full[features].copy()

    X_test = X_test_full[features].copy()

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0) #This was the best
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

    models = [model_1, model_2, model_3, model_4, model_5]

    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)

    for i in range(0, len(models)):
        mae = score_model(models[i])
        print("Model %d MAE: %d" % (i+1, mae))

    # Fill in the best model
    scores = { model:score_model(model) for model in models}
    best_model = min(scores, key=scores.get)
    print(best_model)
    best_model.fit(X, y)
    preds = best_model.predict(X_test)
    output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds})
    output.to_csv('output_basicdataexploration_31.csv', index=False)
    print(output)



def chap32():
    print("2 - Intermediate ML - Missing Values")
    X_full = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./kaggle_house_price_competition/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)
    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)
    
    #REDUCTION
    print("Reduction method")
    reduced_X_train = X_train.copy()
    reduced_X_valid = X_valid.copy()
    missing_values_cols = [col for col in reduced_X_train.columns if reduced_X_train[col].isnull().any()]
    print("Columns with missing values: {}".format(missing_values_cols))
    reduced_X_train = reduced_X_train.drop(missing_values_cols, axis=1)
    reduced_X_valid = reduced_X_valid.drop(missing_values_cols, axis=1)
    print("MAE (REDUCTION METHOD): {}".format(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)))
    
    #IMPUTATION
    print("Imputation method\nI have to make an import on top:\nfrom sklearn.impute import SimpleImputer")
    my_imputer = SimpleImputer(strategy='median')
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) #preprocessing training features
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) #preprocessing validation features
    #imputation removes columns name so I will put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    print("MAE (IMPUTATION METHOD -> median): {}".format(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)))

    final_imputer = SimpleImputer(strategy='mean')
    final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train)) #preprocessing training features
    final_X_valid = pd.DataFrame(final_imputer.transform(X_valid)) #preprocessing validation features
    final_X_test = pd.DataFrame(final_imputer.transform(X_test))
    final_X_train.columns = X_train.columns # put back columns name into X_train
    final_X_valid.columns = X_valid.columns # put back columns name into X_valid

    # Define and fit model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(final_X_train, y_train)
    preds_valid = model.predict(final_X_valid)
    preds_test = model.predict(final_X_test)
    print(len(preds_valid))
    print(len(preds_test))
    print(len(y_valid))
    print(len(y_train))
    #print("MAE {}".format(mean_absolute_error(preds_valid,preds_test))) 

def chap33():
    print("3 - Intermediate ML - Categorical Variables")
    #SETUP
    X = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id') 
    X_test = pd.read_csv('./kaggle_house_price_competition/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll drop columns with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    print(X_train.head())
    #END OF SETUP

    # function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    print("First approach: drop categorical variables")
    drop_X_train = X_train.select_dtypes(exclude='object')
    drop_X_valid = X_valid.select_dtypes(exclude='object')
    print("MAE: {}".format(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)), end="\n\n")

    # .unique() method
    print("Using .unique()")
    print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
    print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique(),  end='\n\n')

    print("Second approach: label encoding")
    # so validation set and training set have different values sets for the same categorical value 
    #   -->delete from dataset categorical columns with different set of values
     
    # take categorical columns
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    # keep only the columns with the same set of values
    good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
    print('Categorical columns that will be label encoded:', good_label_cols)
    print('Categorical columns that will be dropped from the dataset:', bad_label_cols)
    print('\nI\'ve imported LabelEncoder:\n\tfrom sklearn.preprocessing import LabelEncoder')
    # dropping bad columns
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)
    labelEncoder = LabelEncoder()
    for col in good_label_cols:
        label_X_train[col] = labelEncoder.fit_transform(X_train[col])
        label_X_valid[col] = labelEncoder.transform(X_valid[col])
    print("MAE: {}".format(score_dataset(label_X_train, label_X_valid, y_train, y_valid)), end="\n\n")

    #CARDINALIZATION
    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    for i in (sorted(d.items(), key=lambda x: x[1])):  
        print(i)
    print("High cardinality values (more than 10 different values): {}".format([x for x in sorted(d.items(), key=lambda x: x[1]) if x[1]>10]))

    #kaggle way.. keepping columns with less then 10 unique values for one hot encoding
    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('Categorical columns that will be dropped from the dataset:', high_cardinality_cols)

    print("\n\nThird approach: One-Hot encoding")
    print("I've imported OneHotEncoder\n\tfrom sklearn.preprocessing import OneHotEncoder")

    OH_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_enc.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_enc.transform(X_valid[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove ALL categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns (uniqueness < 10) to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    print("MAE: {}".format(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)), end="\n\n")

def chap34():
    print("4 - Intermediate ML - Pipelines")
    print("New imports\n\tfrom sklearn.compose import ColumnTransformer\n\tfrom sklearn.pipeline import Pipeline")

    #SETUP
    X_full = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./kaggle_house_price_competition/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                    train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if
                        X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define model
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                         ])

    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)

    print('Test pipelin created with MAE:', mean_absolute_error(y_valid, preds))
    #END OF SETUP

    #MY PIPELINE
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy="median") # Your code here

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]) # Your code here

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define model
    model = RandomForestRegressor(max_leaf_nodes=720, random_state=200)
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model) ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('My pipeline MAE:', score)
    # Preprocessing of test data, fit model
    preds_test = my_pipeline.predict(X_test)
    print("Prediction o text data\n {}".format(preds_test))

def chap35():
    print("5 -Intermediate ML - Cross Validation\nnew imports:\n\tfrom sklearn.model_selection import cross_val_score\n\timport matplotlib.pyplot as plt")
    #SETUP (CATEGORICAL DATA ARE DROPPED)
    train_data = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')
    test_data = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice              
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Select numeric columns only
    numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[numeric_cols].copy()
    X_test = test_data[numeric_cols].copy()

    #Creating Pipeline with simpleimputer (I got only numeric values) and randomforestregressor
    my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    print("Average MAE score:", scores.mean())
    #END OF SETUP

    def get_score(training_set, training_target, folds, n_estimators):
        #n_estimators -- the number of trees in the forest
        my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
        ])
        scores = -1 * cross_val_score(my_pipeline, training_set, training_target,
                              cv=folds,
                              scoring='neg_mean_absolute_error')
        return scores.mean()

    print("My scoring func MAE mean: {}".format(get_score(X, y, 10, 100)))
    print("test several n_estimators")
    results = {x*50:get_score(X,y,5,x*50) for x in range(1,9)} # Your code here
    print(results)
    best_n_estim=min(results, key=results.get)
    print("Best n estimator: {}".format(best_n_estim))

    
    #use it
    model = RandomForestRegressor(n_estimators=best_n_estim, random_state=0)
    imputer=SimpleImputer()
    imputed_X = pd.DataFrame(imputer.fit_transform(X))
    imputed_X_test = pd.DataFrame(imputer.transform(X_test))
    imputed_X.columns=X.columns
    model.fit(imputed_X,y)
    preds=model.predict(imputed_X_test)
    print(preds)
    print(mean_absolute_error(preds, y))

    #plt.plot(list(results.keys()), list(results.values()))
    #plt.show()

def chap36():
    print("6 - Intermediate ML - XGBoost\nnew imports:\n\tfrom xgboost import XGBRegressor")

    #SETUP
    # Read the data
    X = pd.read_csv('./kaggle_house_price_competition/train.csv', index_col='Id')
    X_test_full = pd.read_csv('./kaggle_house_price_competition/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice              
    X.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                            X_train_full[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # One-hot encode the data (to shorten the code, we use pandas)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)
    #ENDSETUP

    my_model_1 = XGBRegressor(random_state=0) 
    my_model_1.fit(X_train, y_train) 
    predictions_1 = my_model_1.predict(X_valid)
    mae_1 = mean_absolute_error(y_valid, predictions_1)
    print("Mean Absolute Error:" , mae_1)

    #model improvement
    # Define the model
    my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=0)
    my_model_2.fit(X_train, y_train)
    predictions_2 = my_model_2.predict(X_valid)
    mae_2 = mean_absolute_error(predictions_2, y_valid)
    print("Mean Absolute Error: (model improvement)" , mae_2)

    #model deterioration
    my_model_3 = XGBRegressor(n_estimators=10, learning_rate=0.1, random_state=0)
    my_model_3.fit(X_train, y_train)
    predictions_3 = my_model_3.predict(X_valid)
    mae_3 = mean_absolute_error(predictions_3, y_valid)
    print("Mean Absolute Error (model deterioration): " , mae_3)

def chap37():
    print("7 - Intermediate ML - Data Leakage")

    """
    ----------------------The Data Science of Shoelaces-------------------------
    Nike has hired you as a data science consultant to help them save money on 
    shoe materials. 
    Your first assignment is to review a model one of their employees built to 
    predict how many shoelaces they'll need each month. The features going into 
    the machine learning model include:

        1) The current month (January, February, etc)
        2) Advertising expenditures in the previous month
        3)Various macroeconomic features (like the unemployment rate) as of the 
                                                beginning of the current month
        4)The amount of leather they ended up using in the current month

    The results show the model is almost perfectly accurate if you include the 
    feature about how much leather they used. But it is only moderately accurate 
    if you leave that feature out. You realize this is because the amount of 
    leather they use is a perfect indicator of how many shoes they produce, which
    in turn tells you how many shoelaces they need.

    Do you think the leather used feature constitutes a source of data leakage? 
    If your answer is "it depends," what does it depend on?

    SOLUTION:
    This is tricky, and it depends on details of how data is collected (which is 
    common when thinking about leakage). Would you at the beginning of the month 
    decide how much leather will be used that month? If so, this is ok. But if 
    that is determined during the month, you would not have access to it when 
    you make the prediction. 
    If you have a guess at the beginning of the month, and it is subsequently 
    changed during the month, the actual amount used during the month cannot be 
    used as a feature (because it causes leakage).
    
    
    ----------------------Return of the Shoelaces-------------------------------
    You have a new idea. You could use the amount of leather Nike ordered
    (rather than the amount they actually used) leading up to a given month as a
    predictor in your shoelace model.

    Does this change your answer about whether there is a leakage problem? If 
    you answer "it depends," what does it depend on?

    SOLUTION
    This could be fine, but it depends on whether they order shoelaces first or 
    leather first. If they order shoelaces first, you won't know how much leather 
    they've ordered when you predict their shoelace needs. If they order leather 
    first, then you'll have that number available when you place your shoelace 
    order, and you should be ok.

    ----------------------Getting Rich With Cryptocurrencies?-------------------
    You saved Nike so much money that they gave you a bonus. Congratulations.

    Your friend, who is also a data scientist, says he has built a model that 
    will let you turn your bonus into millions of dollars. Specifically, his 
    model predicts the price of a new cryptocurrency (like Bitcoin, but a newer 
    one) one day ahead of the moment of prediction. His plan is to purchase the 
    cryptocurrency whenever the model says the price of the currency (in dollars) 
    is about to go up.

    The most important features in his model are:

        Current price of the currency
        Amount of the currency sold in the last 24 hours
        Change in the currency price in the last 24 hours
        Change in the currency price in the last 1 hour
        Number of new tweets in the last 24 hours that mention the currency
    
    The value of the cryptocurrency in dollars has fluctuated up and down by over  
    𝑖𝑛 𝑡ℎ𝑒 𝑙𝑎𝑠𝑡 𝑦𝑒𝑎𝑟,𝑎𝑛𝑑 𝑦𝑒𝑡 ℎ𝑖𝑠 𝑚𝑜𝑑𝑒𝑙′𝑠 𝑎𝑣𝑒𝑟𝑎𝑔𝑒 𝑒𝑟𝑟𝑜𝑟 𝑖𝑠 𝑙𝑒𝑠𝑠 𝑡ℎ𝑎𝑛 1. He says this is 
    proof his model is accurate, and you should invest with him, buying the 
    currency whenever the model says it is about to go up.

    Is he right? If there is a problem with his model, what is it?

    SOLUTION
    There is no source of leakage here. These features should be available at 
    the moment you want to make a predition, and they're unlikely to be changed 
    in the training data after the prediction target is determined. 
    But, the way he describes accuracy could be misleading if you aren't 
    careful. 
    If the price moves gradually, today's price will be an accurate predictor of
    tomorrow's price, but it may not tell you whether it's a good time to 
    invest. 
    For instance, if it is 𝑡𝑜𝑑𝑎𝑦, 𝑎 𝑚𝑜𝑑𝑒𝑙 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑛𝑔 𝑎 𝑝𝑟𝑖𝑐𝑒 𝑜𝑓 tomorrow may seem 
    accurate, even if it can't tell you whether the price is going up or down 
    from the current price. 
    A better prediction target would be the change in price over the next day. 
    If you can consistently predict whether the price is about to go up or down
    (and by how much), you may have a winning investment opportunity.

    ----------------------Preventing Infections---------------------------------
    An agency that provides healthcare wants to predict which patients from a 
    rare surgery are at risk of infection, so it can alert the nurses to be 
    especially careful when following up with those patients.
    You want to build a model. Each row in the modeling dataset will be a single 
    patient who received the surgery, and the prediction target will be whether 
    they got an infection.

    Some surgeons may do the procedure in a manner that raises or lowers the 
    risk of infection. But how can you best incorporate the surgeon information 
    into the model?
    You have a clever idea.

    Take all surgeries by each surgeon and calculate the infection rate among 
    those surgeons.
    For each patient in the data, find out who the surgeon was and plug in that 
    surgeon's average infection rate as a feature.
    Does this pose any target leakage issues? 
    Does it pose any train-test contamination issues?

    SOLUTION
    This poses a risk of both target leakage and train-test contamination 
    (though you may be able to avoid both if you are careful).
    You have target leakage if a given patient's outcome contributes to the 
    infection rate for his surgeon, which is then plugged back into the 
    prediction model for whether that patient becomes infected. 
    You can avoid target leakage if you calculate the surgeon's infection 
    rate by using only the surgeries before the patient we are predicting for. 
    Calculating this for each surgery in your training data may be a little 
    tricky.

    You also have a train-test contamination problem if you calculate this using
    all surgeries a surgeon performed, including those from the test-set. 
    The result would be that your model could look very accurate on the test set,
    even if it wouldn't generalize well to new patients after the model is 
    deployed. 
    This would happen because the surgeon-risk feature accounts for data in the 
    test set. Test sets exist to estimate how the model will do when seeing new 
    data. So this contamination defeats the purpose of the test set.

    ----------------------Housing Prices----------------------------------------
    You will build a model to predict housing prices. 
    The model will be deployed on an ongoing basis, to predict the price of a 
    new house when a description is added to a website. 
    Here are four features that could be used as predictors.

        1) Size of the house (in square meters)
        2) Average sales price of homes in the same neighborhood
        3) Latitude and longitude of the house
        4) Whether the house has a basement

    You have historic data to train and validate the model.

    Which of the features is most likely to be a source of leakage?

    SOLUTION
    2 is the source of target leakage. 
    Here is an analysis for each feature:
    
    1) The size of a house is unlikely to be changed after it is sold (though 
    technically it's possible). But typically this will be available when we 
    need to make a prediction, and the data won't be modified after the home is 
    sold. So it is pretty safe.

    2) We don't know the rules for when this is updated. 
    If the field is updated in the raw data after a home was sold, and the 
    home's sale is used to calculate the average, this constitutes a case of 
    target leakage. At an extreme, if only one home is sold in the neighborhood, 
    and it is the home we are trying to predict, then the average will be 
    exactly equal to the value we are trying to predict. 
    In general, for neighborhoods with few sales, the model will perform very
    well on the training data. 
    But when you apply the model, the home you are predicting won't have been 
    sold yet, so this feature won't work the same as it did in the training data.

    3) These don't change, and will be available at the time we want to make a 
    prediction. So there's no risk of target leakage here.

    4) This also doesn't change, and it is available at the time we want to 
    make a prediction. So there's no risk of target leakage here.

    """

