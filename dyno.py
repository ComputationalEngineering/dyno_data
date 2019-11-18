# importing depencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import datetime

# unnamed0 = displacement
# unnamed1 = velcoity
# unnamed 2 = force
# unnamed: 3 = dampening

df = pd.read_excel('dynodata.xlsx')
df.drop([0], inplace = True)


df['Unnamed: 0'] = df['Unnamed: 0'].astype(float)
df['Unnamed: 1'] = df['Unnamed: 1'].astype(float)
df['Unnamed: 2'] = df['Unnamed: 2'].astype(float)
df['Unnamed: 3'] = df['Unnamed: 3'].astype(float)




displacement = np.array([df['Unnamed: 0']])
dampening = np.array([df['Unnamed: 3']])


d = {'displacement':displacement, 'dampening':dampening}


dz = pd.DataFrame(data = d)
dz = dz[dz.displacement < .6]
dz = dz[dz.dampening > -300]
z = dz['dampening']
X_train, X_test, y_train, y_test = train_test_split(dz,z,test_size=.2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())




solution = model.fit(X_train,y_train)
predictions = model.predict(X_test)

#plt.scatter(y_test,predictions)
plt.scatter(df['Unnamed: 0'],df['Unnamed: 3'])
plt.xlabel('displacement')
plt.ylabel('dampening')
plt.show()




#y = df['Unnamed: 3']
#y = np.array(y)
#y = sorted(y)
#for i in y:
#    if i < -300:
#        y.remove(i)
#    else:
#        pass

# fitting model restructuring dataframe with new predicted values
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline

#degree = 2
#model = make_pipeline(PolynomialFeatures(degree), LinearRegression())


#model.fit(x,y)
#w = model.predict(x)
#df['Predicted'] = w

# another model
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
#                    hidden_layer_sizes=(5,2), random_state=1)
#clf.fit(x,y)
#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#              beta_1=0.9, beta_2=0.999, early_stopping=False,
#              epsilon=1e-08, hidden_layer_sizes=(5, 2),
#              learning_rate='constant', learning_rate_init=0.001,
#              max_iter=200, momentum=0.9, n_iter_no_change=10,
#              nesterovs_momentum=True, power_t=0.5, random_state=1,
#              shuffle=True, solver='lbfgs', tol=0.0001,
#              validation_fraction=0.1, verbose=False, warm_start=False)
#clf.predict(x)

#plt.plot(x,y)
#plt.plot(x,w)
#plt.show()
#min is a function in python that tells the minimum value for something
# max is a function in python that tells the maximum value for a set of data
