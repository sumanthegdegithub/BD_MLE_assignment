<!-- ABOUT THE PROJECT -->
## About The Project
This project was done as a part of an ML Engineer role take home assignment, Here I have explored ways to process 1B records.
We have trained a model BayesianGaussianMixture on the data and then we have transformed the data to a 2d array, and then we are passing the same 2d arry to get the oriinal data back.

### Approaches tried
1. Converting the BayesianGaussianMixture code to make it dask compatible
    Worked for ~2 days to make the original code from sklearn.mixture compatible with dask, Could not the see the performance that was expected. So scraped the idea
2. Using tensorflow to build BayesianGaussianMixture model. Could not finish either as I did not understand mathematics of
BayesianGaussianMixture completely
3. Batching the data, training BayesianGaussianMixture model with a warm start, working on optimizing other parts of the flow and implementing an early stop. as it was difficult to train the model on whole data and it did not improve after a certain records as the records are duplicated. 

### Create virtual Environment
```
Follow below steps:

$ git clone https://github.com/sumanthegdegithub/BD_MLE_assignment.git
$ cd BD_MLE_assignment
$ python -m venv venv     or  python -m venv <Project_Path>\venv
```

### Run the code
You can run the code from the file test.ipynb