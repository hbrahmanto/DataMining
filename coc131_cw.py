import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy.stats import ttest_rel
from sklearn.manifold import LocallyLinearEmbedding

optimal_hyperparam = {
    'hidden_layer_sizes': (100, 200, 200), 
    'learning_rate_init': 0.001, 
    'alpha': 0.1,
    'max_iter': 300,
    'random_state': 16
}

class COC131:
    def __init__(self):
        self.x = None
        self.y = None
        self.optimal_hyperparam = {}

    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.

        HB: Make sure that the EuroSAT_RGB file is in this coursework file for it to work
        """
        # Setup and initiallization 
        dataset_origin = 'EuroSAT_RGB' # Filename
        image_size = (32, 32) # Image conversion
        
        # Blank arrays to store the image and file labels
        data = [] 
        labels = []

        # Loop function for folders and images
        for f_name in sorted(os.listdir(dataset_origin)): # Gets all the items from EuroSAT_RGB, with each item expected to be a folder representing a file name like AnnualCrop etc.
            f_path = os.path.join(dataset_origin, f_name)
            if not os.path.isdir(f_path): # Skipping anything that its not a folder
                continue

            # Loop function for images in each class folders 
            for image_name in os.listdir(f_path): # Loops only files that ends in .jpg, jpeg or .png and skipping any other files that are not images
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # A code that opens, resize, normalize and store each image to the array
                try:
                    image_path = os.path.join(f_path, image_name)                       # Combines the file path to a class folder with the file name safely 
                    image = Image.open(image_path).convert('RGB').resize(image_size)    # Resizing the image to the desired resolution
                    data.append(np.array(image, dtype=float) / 255.0)                   # Converting the PIL image object into a numpy array and dividing it by 255 so that the neural network can perform better when input
                    labels.append(f_name)

                # An error exception if any image fails to load
                except Exception as e:
                    print(f'Failed to load {image_path}: {e}')
        
        # Once all the data is processed it is then stored in the x and y NumPy arrays
        self.x = np.array(data, dtype=float)  # Stores the images in the array of self.x
        self.y = np.array(labels)             # Stores the labels for each images in the array of self.y

        # Same as the for loop function, but this time if the filename is given, return that specific image
        if filename:
            path = os.path.join(dataset_origin, filename)
            image = Image.open(path).convert('RGB').resize(image_size)

            # Res1 is the flattened array, whilst res2 is the class label extracted from the folder name
            res1 = np.array(image, dtype=float).flatten() / 255.0
            res2 = os.path.basename(os.path.dirname(path))
        
        # If no file name is provided, then it returns an empty value
        else:
            res1 = np.zeros(1)
            res2 = ''

        # Return the result
        return res1, res2


    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.

        HB: This function takes in the inputs of the self values and inp (input Numpy array).
        It then returns the fitted StandardScaler object in res1 and standardized data array with a std dev of 2.5 in res2.
        """
        scaler = StandardScaler()                            # Creating a StandardScaler object that will standardize the data where the mean = 0 and standard deviation = 1
        normalized_value = scaler.fit_transform(inp)         # Applying the scaler value to the input data so that it computes the mean and standard deviation for each feature

        #Rescale to have the standard deviation of 2.5 as the default value is 1
        res1 = scaler                                        
        res2 = normalized_value * 2.5                        # Multiplying the standardized value by 2.5 as specified

        # Returns the StandardScaler in res1 and the rescaled value of the standard deviation in res2
        return res1, res2 

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found. 

        HB: When this trainer is put to the test in the jupyter notebook, the value of hidden_layer_size (64) was one of the first 
        values to be tested. This is beacuse, it is a common number for it start in and it achieved decent accuracy for both testing and training.
        The value was increased to (100, 100) with also an additional layer to achieve better accuracy.
        """
        # Standardize the data and flatten from a 4D array
        x_flattened_value = self.x.reshape(len(self.x), -1)     # As self.x is a 4D array it needs to be converted back into a 1D array of 3072 values
        res1, res2 = self.q2(x_flattened_value)                 # Assigning the value of res2 for the standardized value that is run through the function of question 2
        x_flattened_value = res2                                # Re-assigning the standardized value from res2 for a ready to train data
        y = self.y                                              # Holding the values of the labels
        
        # Data splitting
        if pre_split_data:
            train_x, test_x, train_y, test_y = pre_split_data                                 # Use pre-split values if its given
        elif test_size is not None:                                                           # If test_size is given, perform a new split
            train_x, test_x, train_y, test_y = train_test_split(                              # Split the data into training and testing sets
                x_flattened_value, y, test_size=test_size, stratify=y, random_state=16        # Use stratified split for balance class distribution and set a random seed for reepoducibility
            )
        else:
            raise ValueError('Either test_size must be set or pre_split_data must be given.')
        
        # Set default hyperparameters if none is provided
        if hyperparam is None:
            hyperparam = {
                'hidden_layer_sizes': [(100,)],                # Setting how much neurons and layers for model training
                'learning_rate_init': [0.001],                 # Setting the initial learning rate for model training
                'alpha': [0.0001]                              # Setting the regularization for the model on how complex it can learn the patterns
            }
        
        # Initialize tracking
        best_model = None                                      # A placeholder that indicates the best performing MLPClassifier
        best_testing_accuracy = 0                              # A placeholder that will be used to compare the final test accuracies, hence it will only update if there are better scores
        best_loss = []                                         # An array that stores the loss curve (loss values across training iterations) for the best model
        best_training_accuracy = []                            # An array that stroes the training accuracy curve for the best model
        best_testing_accuracy_list = []                        # An array that stores the testing accuracy curve for the best model

        # Trying out every combination of hyperparameters
        for hidden_layers in hyperparam['hidden_layer_sizes']:             # Taking every test value for hidden layer from the array of hyperparam that is set in the notebook
            for learning_rate in hyperparam['learning_rate_init']:         # Followed by learning rate
                for alpha_ in hyperparam['alpha']:                         # and alpha value

                    # Creating the neural network with this setup
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,                  # It is then passed through the MLPClassifier model for it to learn with the data
                        learning_rate_init=learning_rate,
                        alpha=alpha_,
                        max_iter=300,                                      # Setting the iteration value on how much data it can take
                        random_state=16,                                   # Random state is used to control the randomness in functions that involve anny kind of random process
                        verbose=False                                   
                    )

                    # Training the model with the training data
                    model.fit(train_x, train_y)                                                   # Fitting the MLP model using the training features and labels

                    # Tracking how the model performed throughout the training
                    loss_during_training = model.loss_curve_                                      # Stores the loss values after each iteration during training for plots

                    # Calculating the accuracy for both training and testing data
                    training_accuracy = accuracy_score(train_y, model.predict(train_x))           
                    testing_accuracy = accuracy_score(test_y, model.predict(test_x))              

                    # Creating lists to match the length of the loss curve (for plotting)
                    training_accuracy_curve = [training_accuracy] * len(loss_during_training)    
                    testing_accuracy_curve = [testing_accuracy] * len(loss_during_training)

                    # Statement that compares every other model and saves the best information
                    if testing_accuracy_curve[-1] > best_testing_accuracy:                       # Checks if this model has the best test accuracy so far
                        best_model = model                                                       # If it does it saves the best model found
                        best_testing_accuracy = testing_accuracy                                 # Update the best testing accuracy value
                        best_loss = loss_during_training                                         # Store the loss values of the best model
                        best_training_accuracy = training_accuracy_curve                         # Store the repeated training accuracy values
                        best_testing_accuracy_list = testing_accuracy_curve                      # Store the repeated testing accuracy values

                        # Save the best parameters
                        self.optimal_hyperparam = {
                            'hidden_layer_sizes': hidden_layers,
                            'learning_rate_init': learning_rate,
                            'alpha': alpha_
                        }
        
        return best_model, np.array(best_loss), np.array(best_training_accuracy), np.array(best_testing_accuracy_list)
    
    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """
        # List of alpha values to test
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

        # Get the best hyperparameter from the training the model in q3
        best_param = optimal_hyperparam

        # Standardize the data and flatten from a 4D array
        x_flattened_value = self.x.reshape(len(self.x), -1)     # As self.x is a 4D array it needs to be converted back into a 1D array of 3072 values
        res1, res2 = self.q2(x_flattened_value)                 # Assigning the value of res2 for the standardized value that is run through the function of question 2
        x_flattened_value = res2                                # Re-assigning the standardized value from res2 for a ready to train data
        y = self.y                                              # Holding the values of the labels
        
        # Data split
        train_x, test_x, train_y, test_y = train_test_split(
            x_flattened_value, y, test_size=0.3, stratify=y, random_state=16
        )

        result = []

        # Train model for each alpha and apply the test accuracy
        for alpha_value in alpha_values:
            try:
                print(f'Training model with alpha = {alpha_value}')          # Displaying the current value being used for training

                #Using the optimal hyperparam values for the MLPClassifier
                model = MLPClassifier(
                    hidden_layer_sizes=best_param['hidden_layer_sizes'],
                    learning_rate_init=best_param['learning_rate_init'],
                    alpha=alpha_value,                                       # Alpha value differs as it is looping through the assigned values to find the most accurate one
                    max_iter=best_param['max_iter'],
                    random_state=best_param['random_state']
                )

                model.fit(train_x, train_y)                                  # Training the model with current alpha
                predicted_y = model.predict(test_x)                          # Predicting the test set outcomes unig the trained model
                accuracy = accuracy_score(test_y, predicted_y)               # Calculating accuracy for the test set
                result.append(accuracy)                                      # Storing the accuracy result for this alpha value

            except Exception as e:
                print(f'Failed for alpha={alpha_value}: {e}')                # Prints error message for debugging
                result.append(0)

        return np.array(result)                                              # Returns all accuracy results as a NumPy array

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """
        # Get the best hyperparameter from the training the model in q3
        best_param = optimal_hyperparam

        # Standardize the data and flatten from a 4D array
        x_flattened_value = self.x.reshape(len(self.x), -1)     # As self.x is a 4D array it needs to be converted back into a 1D array of 3072 values
        res1, res2 = self.q2(x_flattened_value)                 # Assigning the value of res2 for the standardized value that is run through the function of question 2
        x_flattened_value = res2                                # Re-assigning the standardized value from res2 for a ready to train data
        y = self.y                                              # Holding the values of the labels

        # Creating the MLPClassifier using the best parameters
        model = MLPClassifier(
            hidden_layer_sizes=best_param['hidden_layer_sizes'],
            learning_rate_init=best_param['learning_rate_init'],
            alpha=best_param['alpha'],
            max_iter=best_param['max_iter'],
            random_state=best_param['random_state']
        )

        # 5-Fold CV with stratification
        stratify = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)                       # Ensures that each fold has the same class distribution as the original dataset
        stratify_scores = cross_val_score(model, x_flattened_value, y, cv=stratify)                 # Runs 5-fold cross-validation using stratified folds and stores the accuracy scores

        # 5-Fold CV without stratification
        non_stratify = KFold(n_splits=5, shuffle=True, random_state=16)                             # Regular KFold without class balancing
        non_stratify_scores = cross_val_score(model, x_flattened_value, y, cv=non_stratify)         # Runs 5-fold cross-validation using non-stratified folds and stores the accuracy scores

        # Hypothesis test (paired t-test)
        p_value = ttest_rel(stratify_scores, non_stratify_scores).pvalue

        # Decision based on p-values (alpha = 0.1)
        if p_value < 0.1:                                                                           # If p-value is less than 0.1, then the difference between the two methods is statistically significanat
            conclusion = 'Splitting method impacted performance'
        else:
            conclusion = 'Splitting methtod had no effect'

        # Return the 4 required result
        res1 = np.mean(stratify_scores)
        res2 = np.mean(non_stratify_scores)
        res3 = p_value
        res4 = conclusion

        return res1, res2, res3, res4, stratify_scores, non_stratify_scores

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.

        HB: n_neighbors were adjusted accordingly when the function is ran in jupyter notebook for data plotting as some came out un-readable.
        """
        # Standardize the data and flatten from a 4D array
        x_flattened_value = self.x.reshape(len(self.x), -1)     # As self.x is a 4D array it needs to be converted back into a 1D array of 3072 values

        # Applying the imported Locally Linear Embedding to reduce the data to 2D
        local_lin_emb = LocallyLinearEmbedding(n_components=2, n_neighbors=30, random_state=16)    # Initializing LLE to reduce the HD image data into 2D using 30 neighbors for local structure
        res = local_lin_emb.fit_transform(x_flattened_value)

        return res