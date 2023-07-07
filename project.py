import pandas as pd

df = pd.read_csv("50_Startups.csv")

print(df) # prints all the data

#creating dummy variables
dumm=pd.get_dummies(df.State)
print(dumm) # it will print the number of states present in the data file

# now joining the two dataframes:
m = pd.concat([df, dumm], axis='columns')
print(m)

#now we can drop the State column: 
#and also we have to drop one of the new columns because it can interfere with our ML model, i.e., it can interfere with dummy variable track

# so we are left with one less col: lets drop florida

new = m.drop(['State', 'Florida'], axis='columns')

print(new) # printing the new dataframe

from sklearn.linear_model import LinearRegression()
model = LinearRegression()

x = new.drop('Profit', axis='coolumns')
print(x)

y = new.Marketing
print(y)

model.fit(x, y)
