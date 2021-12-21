import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

def chap1():
    #1  Hello, Seaborn!
    #LOAD DATA
    # Read the file into a variable fifa_data
    path_fifacsv = "./dataviz_files/fifa.csv"
    fifa_data = pd.read_csv(path_fifacsv, index_col="Date", parse_dates=True)
    plt.figure(figsize=(16,6))
    print(fifa_data.head())
    # Line chart showing how FIFA rankings evolved over time
    sns.lineplot(data=fifa_data)
    plt.show()
    
def test():
    sns.set(style='whitegrid')
    tips = sns.load_dataset('tips')
    ax = sns.boxplot(x=tips['total_bill'])
    plt.show()

def chap2():
    #2 Line charts
    museum_filepath = "./dataviz_files/museum_visitors.csv"
    museum_data = pd.read_csv(museum_filepath, index_col="Month", parse_dates=True)
    # Print the last five rows of the data 
    print(museum_data.tail())
    sns.lineplot(data=museum_data)
    plt.show()
    # Line plot showing the number of visitors to Avila Adobe over time
    plt.figure(figsize=(12,6))
    plt.title("Monthly Visitors to Avila Adobe")
    sns.lineplot(data=museum_data['Avila Adobe'], label = "Visitors") # Your code here
    plt.xlabel("Month")
    plt.show()

def chap3():
    #3 BAR CHART + HEATMAPS
    # Path of the file to read
    ign_filepath = "./dataviz_files/gamedata.csv"

    # Fill in the line below to read the file into a variable ign_data
    ign_data = pd.read_csv(ign_filepath, index_col="Platform")
    plt.figure(figsize=(20,6))
    sns.barplot(x=ign_data.index, y=ign_data['Racing'])
    plt.show()

    # Heatmap showing average game score by platform and genre
    plt.figure(figsize=(10,10))
    sns.heatmap(ign_data, annot=True)
    # Add label for horizontal axis
    plt.xlabel("Genre")
    # Add label for vertical axis
    plt.title("Average Game Score, by Platform and Genre")
    plt.show()

def chap4():
    #4 Scatter plots
    candy_data=pd.read_csv('./dataviz_files/candy.csv', index_col='id')
    #Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
    print(candy_data)
    sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
    plt.show()
    sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
    plt.show()
    sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])
    plt.show()
    sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)
    plt.show()
    # categorical Scatter plot showing the relationship between 'chocolate' and 'winpercent'
    sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])
    plt.show()

def chap5_6():
    #5 Distributions
    iris_data = pd.read_csv('./dataviz_files/Iris.csv', index_col='Id')
    print(iris_data.head())
    #histogram distplot deprecated --> histplot or displot
    sns.distplot(a=iris_data['PetalLengthCm'], kde=False)
    plt.show()
    sns.displot(x=iris_data['PetalLengthCm'], y=iris_data['SepalLengthCm'], kind='hist')
    plt.show()
    sns.histplot(x=iris_data['PetalLengthCm'], y=iris_data['SepalLengthCm'])
    plt.show()
    #densityplot
    sns.kdeplot(data=iris_data['PetalLengthCm'], shade=True)
    plt.show()
    #2d kde plot 
    sns.jointplot(x=iris_data['PetalLengthCm'], y=iris_data['SepalLengthCm'], kind="kde")
    plt.show()
    # Paths of the files to read
    iris_setosa = iris_data.query("Species=='Iris-setosa'")
    iris_versicolor = iris_data.query("Species=='Iris-versicolor'")
    iris_virginica = iris_data.query("Species=='Iris-virginica'")
    
    sns.kdeplot(data=iris_setosa['PetalLengthCm'], label="Iris-setosa", shade=True)
    sns.kdeplot(data=iris_versicolor['PetalLengthCm'], label="Iris-versicolor", shade=True)
    sns.kdeplot(data=iris_virginica['PetalLengthCm'], label="Iris-virginica", shade=True)
    plt.title("Distribution of Petal Lengths, by Species")
    plt.legend()
    plt.show()

    # Fill in the line below: In the first five rows of the data for benign tumors, what is the
    # largest value for 'Perimeter (mean)'?
    # max_perim = max(cancer_b_data.head()['Perimeter (mean)'])

    """
    Trends - A trend is defined as a pattern of change.
    sns.lineplot - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.

    Relationship - There are many different chart types that you can use to understand relationships between variables in your data.
    sns.barplot - Bar charts are useful for comparing quantities corresponding to different groups.
    sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.
    sns.scatterplot - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
    sns.regplot - Including a regression line in the scatter plot makes it easier to see any linear relationship between two variables.
    sns.lmplot - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
    sns.swarmplot - Categorical scatter plots show the relationship between a continuous variable and a categorical variable.

    Distribution - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
    sns.distplot - Histograms show the distribution of a single numerical variable.
    sns.kdeplot - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
    sns.jointplot - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.   



    STYLES
    sns.styles("stylename")
        "darkgrid"
        "whitegrid"
        "dark"
        "white"
        "ticks"
    """

def chap6()
    return

chap5()