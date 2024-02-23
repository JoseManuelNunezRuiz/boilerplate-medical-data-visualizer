import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column

df['overweight'] = df.apply(lambda row: 1 if (row['weight'] / ((row['height']/100)**2))  > 25 else 0, axis = 1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = df.apply(lambda row: 0 if row['cholesterol'] == 1 else 1, axis = 1)
df['gluc'] = df.apply(lambda row: 0 if row['gluc'] == 1 else 1, axis = 1)

# Draw Categorical Plot
def draw_cat_plot():    
    
    # Crear un nuevo DataFrame con las columnas seleccionadas
    df_catplot = df.melt(id_vars='cardio', value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], value_name='value')

    df_catplot = pd.DataFrame({'total':df_catplot.groupby(['cardio', 'variable'])['value'].value_counts()}).rename(columns={'cardio':'Cardio','variable':'Variable', 'value':'Value'}).reset_index()
    catplot = sns.catplot(data=df_catplot, x='variable', y='total', col='cardio', kind='bar', hue='value')


    # Get the figure for the output
    fig = catplot.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    
    df = df[df['ap_lo'] <= df['ap_hi']]

    # Clean
    
    height_lower_threshold = df['height'].quantile(0.025)
    height_upper_threshold = df['height'].quantile(0.975)
    
    df = df[(df['height'] >= height_lower_threshold) & (df['height'] <= height_upper_threshold)]
    
    weight_lower_threshold = df['weight'].quantile(0.025)
    weight_upper_threshold = df['weight'].quantile(0.975)
    
    df_heat = df[(df['weight'] >= weight_lower_threshold) & (df['weight'] <= weight_upper_threshold)]

    # Calculate the correlation matrix
    
    corr = round(df_heat.corr(),1)

    # Generate a mask for the upper triangle
    
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # Set up the matplotlib figure
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix (Upper Triangle)')


    # Do not modify the next two lines
    
    fig.savefig('heatmap.png')
    return fig
