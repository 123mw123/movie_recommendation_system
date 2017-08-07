from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize

data = loadmat('ex8_movies.mat')
#print data

#movies-users rating matrix

Y_Ratings = data["Y"]

#movies-users rating matrix with elements equal to 1 when movie is rated else zero

R_Bool = data["R"]

print Y_Ratings.shape, R_Bool.shape


parameters_data = loadmat('ex8_movieParams.mat')

#dataset
X = parameters_data['X']
Theta = parameters_data['Theta']
print X.shape, Theta.shape


movies = Y_Ratings.shape[0]
users = Y_Ratings.shape[1]
features = 10
learning_rate = 10

#intialising parameters with random values

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

print X.shape, Theta.shape, params.shape


mean_Y = np.zeros((movies, 1))
normal_Y = np.zeros((movies, users))

#"normalising ratings"

for i in range(movies):
    idx = np.where(R_Bool[i,:] == 1)[0]
    mean_Y[i] = Y_Ratings[i,idx].mean()
    normal_Y[i, idx] = Y_Ratings[i, idx] - mean_Y[i]

#building cost function

def cost(params, Y, R, number_of_features, learning_rate):


    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * number_of_features], (num_movies, number_of_features)))
    Theta = np.matrix(np.reshape(params[num_movies * number_of_features:], (num_users, number_of_features)))


    error = np.multiply((X * Theta.T) - Y, R)
    squared_error = np.power(error, 2)
    J = (1. / 2) * np.sum(squared_error)+ ((learning_rate / 2) * np.sum(np.power(Theta, 2)))\
        + ((learning_rate / 2) * np.sum(np.power(X, 2)))


    X_gradient = (error * Theta) + (learning_rate * X)
    Theta_gradient = (error.T * X) + (learning_rate * Theta)


    grad = np.concatenate((np.ravel(X_gradient), np.ravel(Theta_gradient)))

    return J, grad


#minimizing cost function
fmin = minimize(fun=cost, x0=params, args=(normal_Y, R_Bool, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})
print fmin

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

print X.shape, Theta.shape

predictions = X * Theta.T
my_preds = predictions[:, -1] + mean_Y
print my_preds.shape


movie_idx = {}
file = open('movie_ids.txt')
for line in file:
    tokens = line.split(' ')
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
idx = np.argsort(my_preds, axis=0)[::-1]

#changing ratings into integers between 0-5
for i in range(my_preds.shape[0]):
    my_preds[i] = int(my_preds[i])
    if int(my_preds[i]) > 5:
        my_preds[i] = 5
    elif int(my_preds[i]) < 0:
        my_preds[i] = 0

print("movie predictions:")
for i in range(my_preds.shape[0]):
    j = int(idx[i])
    print int(my_preds[j]), movie_idx[j]
