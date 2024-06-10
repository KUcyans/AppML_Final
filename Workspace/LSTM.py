import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Layer, LayerNormalization, LSTM, Dense, Dropout, Masking, Concatenate, Input, Activation, Lambda
from sklearn.model_selection import KFold
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from tqdm import tqdm
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import Preprocess as pp

'''
`divideInputAndTarget(sequence)`
Divides a sequence into input and target sequences for model training.
- **Parameters:**
  - `sequence` (list/ndarray): Input sequence to be divided.
- **Returns:**
  - `input_seq` (ndarray): Sequence excluding the last element.
  - `target_seq` (ndarray): Sequence excluding the first element.
'''

def divideInputAndTarget(sequence):
    input_seq = sequence[:-1]
    target_seq = sequence[1:]
    return np.array(input_seq), np.array(target_seq)

'''
`getPaddedSequence(sequence, maxNplays, padding_value=0)`
Pads a sequence to a specified length with a given padding value.
- **Parameters:**
  - `sequence` (list/ndarray): Sequence to be padded.
  - `maxNplays` (int): Desired length of the padded sequence.
  - `padding_value` (int, optional): Value to pad the sequence with. Default is 0.
- **Returns:**
    - `padded` (ndarray): Padded sequence.
'''

def getPaddedSequence(sequence, maxNplays, padding_value=0):
    padded = np.full((maxNplays, len(sequence[0])), padding_value)
    padded[:len(sequence)] = sequence
    return padded

'''
`getPaddedTarget(sequence, maxNplays, padding_value=0)`
Pads a target sequence to a specified length with a given padding value.
- **Parameters:**
- `sequence` (list/ndarray): Target sequence to be padded.
- `maxNplays` (int): Desired length of the padded sequence.
- `padding_value` (int, optional): Value to pad the sequence with. Default is 0.
- **Returns:**
- `padded` (ndarray): Padded target sequence.
'''

def getPaddedTarget(sequence, maxNplays, padding_value=0):
    padded = np.full(maxNplays, padding_value)
    padded[:len(sequence)] = sequence
    return padded

'''
`plotActivations()`
Plots various activation functions.
- **No Parameters.**
- **No Returns.**
- **Description:**
- Plots the following activation functions: Softplus, Tanh, ReLU, Leaky ReLU, PReLU, ELU, SELU, Sigmoid, Hard Sigmoid, Hard Tanh.
'''
def plotActivations():
    def softplus(x):
        return np.log(1 + np.exp(x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    def prelu(x, alpha=0.1):
        return np.where(x > 0, x, x * alpha)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def selu(x):
        alpha = 1.67326
        scale = 1.0507
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def hard_sigmoid(x):
        return np.clip(0.2 * x + 0.5, 0, 1)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def hard_tanh(x):
        return np.clip(x, -1, 1)
    
    activationFunctions = [
        (softplus, 'Softplus'), 
        (tanh, 'Tanh'), 
        (relu, 'ReLU'), 
        (lambda x: leaky_relu(x, alpha=0.01), 'Leaky ReLU'), 
        (lambda x: prelu(x, alpha=0.1), 'PReLU'), 
        (lambda x: elu(x, alpha=1.0), 'ELU'), 
        (selu, 'SELU'), 
        (sigmoid, 'Sigmoid'), 
        (hard_sigmoid, 'Hard Sigmoid'), 
        (hard_tanh, 'Hard Tanh')
    ]

    x = np.linspace(-10, 10, 75)

    fig, ax = plt.subplots(figsize=(12, 8))
    for func, name in activationFunctions:
        y = func(x)
        ax.plot(x, y, label=f'{name} Activation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Activation Functions')
    ax.legend()
    plt.grid(True)
    plt.show()

'''
`buildLSTMCore(Nunits, Nfeatures, outputClassification, outputRegression, activators)`
Builds an LSTM model with specified parameters.
- **Parameters:**
  - `Nunits` (int): Number of units in the LSTM layer.
  - `Nfeatures` (int): Number of features in the input data.
  - `outputClassification` (dict): Dictionary of classification output layers.
  - `outputRegression` (dict): Dictionary of regression output layers.
  - `activators` (dict): Dictionary of activation functions for different layers.
- **Returns:**
  - `model` (tf.keras.Model): Compiled LSTM model.
  '''

def buildLSTMCore(Nunits, Nfeatures, outputClassification, outputRegression, activators):
    inputs = Input(shape=(None, Nfeatures))
    x = Masking(mask_value=0)(inputs)
    x = LSTM(units=Nunits, activation=activators['LSTM'], return_sequences=True)(x)

    outputs = []

    for output_name, n_units in outputClassification.items():
        outputs.append(Dense(n_units, activation=activators['classification'], name=output_name)(x))

    for output_name in outputRegression.keys():
        if output_name in ['gameClockSecondsExpired', 'gameClockStoppedAfterPlay']:
            outputs.append(Dense(1, activation=activators['clocks'], name=output_name)(x))
        else:
            outputs.append(Dense(1, activation=activators['yards'], name=output_name)(x))

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    losses = {name: 'categorical_crossentropy' for name in outputClassification}
    losses.update({name: 'mean_squared_error' for name in outputRegression})

    metrics = {name: 'accuracy' for name in outputClassification}
    metrics.update({name: 'mean_squared_error' for name in outputRegression})

    model.compile(optimizer=Adam(), loss=losses, metrics=metrics)
    return model

'''
`trainLSTM(NFLplays_split_train, activators, fileName, isDebug=False, isSave=False)`
Trains the LSTM model on provided training data.
- **Parameters:**
  - `NFLplays_split_train` (list): List of training sequences.
  - `activators` (dict): Dictionary of activation functions for different layers.
  - `fileName` (str): Name of the file to save the model.
  - `isDebug` (bool, optional): Whether to print debug information. Default is False.
  - `isSave` (bool, optional): Whether to save the trained model. Default is False.
- **Returns:**
  - `model` (tf.keras.Model): Trained LSTM model.
  - `scalers` (dict): Dictionary of scalers used for regression features.
'''
def trainLSTM(NFLplays_split_train, activators, fileName=f"LSTM_{datetime.today().strftime('%d%m%y')}.keras", isDebug=False, isSave=False):
    # -------- tune these parameters or not --------
    Nunits = 100
    Nbatch = 128
    # ---------------------------------------------
    
    maxNplays = max(len(game) for game in NFLplays_split_train)
    Nfeatures = len(pp.getColumns('playCircumstance'))
    
    classification_features = ['playType', 'huddle', 'formation', 'playResult', 'noPlay']
    regression_features = ['gameClockSecondsExpired', 'gameClockStoppedAfterPlay', 'offensiveYards']
    
    outputClassification = {col: len(np.unique(np.concatenate([game[col].values for game in NFLplays_split_train]))) for col in classification_features}
    outputRegression = {col: 1 for col in regression_features}

    if isDebug:
        print("Output classification classes:", outputClassification)
        print("Output regression classes:", outputRegression)

    model = buildLSTMCore(Nunits, Nfeatures, outputClassification, outputRegression, activators)
    
    X_train = []
    y_train_dict = {col: [] for col in classification_features + regression_features}
    
    scalers = {col: RobustScaler() for col in regression_features}
    
    for game in tqdm(NFLplays_split_train, desc="Preparing data"):
        inputSequence = []
        targetSequences = {col: [] for col in classification_features + regression_features}
        
        for _, play in game.iterrows():
            play_input = pp.getCircumstances(play).values
            inputSequence.append(play_input)
            for col in classification_features + regression_features:
                targetSequences[col].append(play[col])
        
        X_train.append(getPaddedSequence(inputSequence, maxNplays))
        for col in classification_features + regression_features:
            y_train_dict[col].append(getPaddedTarget(targetSequences[col], maxNplays))
    
    X_train = np.array(X_train)
    for col in classification_features + regression_features:
        y_train_dict[col] = np.array(y_train_dict[col])
    
    encoder_dict = {}
    
    for col in classification_features:
        encoder = OneHotEncoder(sparse_output=False)
        y_train_dict[col] = encoder.fit_transform(y_train_dict[col].reshape(-1, 1)).reshape(X_train.shape[0], maxNplays, -1)
        encoder_dict[col] = encoder
        if isDebug:
            print(f"{col} unique categories:", encoder.categories_)
    
    for col in regression_features:
        flat_targets = y_train_dict[col].reshape(-1, 1)
        scaled_targets = scalers[col].fit_transform(flat_targets).reshape((X_train.shape[0], maxNplays, 1))
        y_train_dict[col] = scaled_targets
    
    if isDebug:
        print(f"X_train shape: {X_train.shape}")
        for col in classification_features + regression_features:
            print(f"y_train_dict[{col}] shape: {y_train_dict[col].shape}")
        print(f"Keys in training data dictionary: {list(y_train_dict.keys())}")
        print(f"Model output names: {model.output_names}")
    
    model.fit(X_train, y_train_dict, epochs=10, batch_size=Nbatch, validation_split=0.2, verbose=1)
    if isSave:
        dirPath = '../LSTM/Models/'
        model.save(dirPath + fileName)
    return model, scalers

'''
`loadModel(fileName)`
Loads a saved LSTM model from file.
- **Parameters:**
  - `fileName` (str): Name of the file containing the saved model.
- **Returns:**
  - `model` (tf.keras.Model): Loaded LSTM model.
'''
def loadModel(fileName):
    dirPath = '../LSTM/'
    return load_model(dirPath+fileName)
'''
`loadScaler(fileName)`
Loads a saved scaler from file.
- **Parameters:**
  - `fileName` (str): Name of the file containing the saved scaler.
- **Returns:**
  - `scaler` (sklearn.preprocessing): Loaded scaler.
'''
def loadScaler(fileName):
    dirPath = '../LSTM/'
    return joblib.load(dirPath+fileName)


'''
 `buildExtendedDF(actual_dict, predicted_dict)`
Builds a DataFrame comparing actual and predicted values.
- **Parameters:**
  - `actual_dict` (dict): Dictionary of actual values.
  - `predicted_dict` (dict): Dictionary of predicted values.
- **Returns:**
  - `dfs` (dict): Dictionary of DataFrames comparing actual and predicted values.
'''
def buildExtendedDF(actual_dict, predicted_dict):
    dfs = {}

    for feature in actual_dict.keys():
        actual = actual_dict[feature].astype(float)
        predicted = predicted_dict[feature].astype(float)

        max_length = max(len(actual), len(predicted))

        actual_padded = np.pad(actual, (0, max_length - len(actual)), constant_values=np.nan)
        predicted_padded = np.pad(predicted, (0, max_length - len(predicted)), constant_values=np.nan)

        df = pd.DataFrame({'Actual': actual_padded, 'Predicted': predicted_padded})
        
        if feature == 'gameClockSecondsExpired':
            df['Cumulative_Actual'] = np.cumsum(np.nan_to_num(df['Actual']))
            df['Cumulative_Predicted'] = np.cumsum(np.nan_to_num(df['Predicted']))
        
        dfs[feature] = df

    return dfs
'''
`predictExtendedLSTM(model, scalers, testGame, isDebug=False)`
Makes predictions using the trained LSTM model and builds a comparison DataFrame.
- **Parameters:**
  - `model` (tf.keras.Model): Trained LSTM model.
  - `scalers` (dict): Dictionary of scalers used for regression features.
  - `testGame` (pd.DataFrame): DataFrame of test game data.
  - `isDebug` (bool, optional): Whether to print debug information. Default is False.
- **Returns:**
  - `dfs` (dict): Dictionary of DataFrames comparing actual and predicted values.
  - `Nplays` (int): Number of plays predicted.
'''
def predictExtendedLSTM(model, scalers, testGame, isDebug=False):
    inputSequence = np.array([pp.getCircumstances(play).values for index, play in testGame.iterrows()])
    inputSequence = np.expand_dims(inputSequence, axis=0)
    
    if isDebug:
        print(f'Initial inputSequence shape: {inputSequence.shape}')

    predicted_dict = {name: [] for name in model.output_names}
    current_input = inputSequence
    accumulated_seconds = 0

    Nplays = 0
    while accumulated_seconds <= 3600:
        predictions = model.predict(current_input, verbose=0)
        
        for name, pred in zip(model.output_names, predictions):
            if pred.shape[-1] > 1:
                predicted_value = np.argmax(pred, axis=-1).flatten()[-1]
            else:
                predicted_value = pred.flatten()[-1]
                if name in scalers:
                    predicted_value = scalers[name].inverse_transform([[predicted_value]])[0, 0]
            predicted_dict[name].append(predicted_value)
        elapsedTime = predicted_dict['gameClockSecondsExpired'][-1]
        accumulated_seconds += elapsedTime
        if isDebug:
            print(f'elapsedTime: {elapsedTime}')

        new_play = np.zeros((1, 1, current_input.shape[2]))
        for j, name in enumerate(model.output_names):
            if j < current_input.shape[2]:
                predicted_value = predicted_dict[name][-1]
                new_play[0, 0, j] = predicted_value
        
        current_input = np.concatenate([current_input[:, 1:, :], new_play], axis=1)
        Nplays += 1
    actual_dict = {name: testGame[name].values for name in model.output_names}

    predicted_dict = {name: np.array(predicted_dict[name]) for name in model.output_names}

    return buildExtendedDF(actual_dict, predicted_dict), Nplays

'''
`getActivationSummary(debugTrain, debugTest, isDebug=False)`
Trains and evaluates the model using different activation functions for clock-related features.
- **Parameters:**
  - `debugTrain` (list): List of training sequences for debugging.
  - `debugTest` (list): List of test sequences for debugging.
  - `isDebug` (bool, optional): Whether to print debug information. Default is False.
- **Returns:**
  - `results` (dict): Dictionary of results for each activation function.
'''
def getActivationSummary(debugTrain, debugTest, clock_activations, isDebug=False):
    results = {}

    for activation in clock_activations:
        activators = {
            'LSTM': 'tanh',
            'classification': 'softmax',
            'clocks': activation,
            'yards': 'relu',
        }
        print(f"Training with clocks activation: {activation}")

        start_time = time.time()
        model_test, scaler_test = trainLSTM(debugTrain, activators, 'test.keras', isDebug=isDebug, isSave=False)
        train_time = time.time()
        
        training_duration = train_time - start_time
        print(f"Training time for {activation}: {training_duration:.2f} seconds")

        prediction_0, Nplays_0 = predictExtendedLSTM(model_test, scaler_test, debugTest[0], isDebug=isDebug)
        predict_time = time.time()

        prediction_duration = predict_time - train_time
        print(f"Prediction time for {activation}: {prediction_duration:.2f} seconds")

        cumulative_predicted = prediction_0['gameClockSecondsExpired'].sum()
        cumulative_actual = debugTest[0]['gameClockSecondsExpired'].sum()

        val_loss = prediction_0['val_loss'][-1]  # Assuming 'val_loss' is part of prediction_0
        
        print(f"Validation loss for {activation}: {val_loss:.4f}")

        results[activation] = {
            'model': model_test,
            'scaler': scaler_test,
            'prediction': prediction_0,
            'Nplays': Nplays_0,
            'train_time': training_duration,
            'predict_time': prediction_duration,
            'val_loss': val_loss,
            'cumulative_predicted': cumulative_predicted,
            'cumulative_actual': cumulative_actual
        }

    print("\n# Model Performance Summary\n")
    for activation, result in results.items():
        print(f"## {activation.capitalize()} Activation:")
        print(f"- **Training Time:** {result['train_time']:.2f} seconds")
        print(f"- **Prediction Time:** {result['predict_time']:.2f} seconds")
        print(f"- **Val Loss (Epoch 10):** {result['val_loss']:.4f}")
        print(f"- **Predictions:**")
        print(f"  - **Cumulative Predicted:** {result['cumulative_predicted']:.2f}")
        print(f"  - **Cumulative Actual:** {result['cumulative_actual']:.2f}")
        print()

    return results

'''
`runPrediction(model, scaler, testDataList, isDebug=False)`
Runs predictions on a list of test games using the trained LSTM model.
- **Parameters:**
  - `model` (tf.keras.Model): Trained LSTM model.
  - `scaler` (sklearn.preprocessing): Scaler used for regression features.
  - `testDataList` (list): List of test game data.
  - `isDebug` (bool, optional): Whether to print debug information. Default is False.
- **Returns:**
  - `myPrecious` (list): List of DataFrames containing prediction results for each test game.
'''
def savePredictionCore(model, scaler, testGame, i, dirPath, isDebug):
    prediction_df, _ = predictExtendedLSTM(model, scaler, testGame, isDebug=isDebug)
    
    combined_df = pd.concat(prediction_df.values(), axis=1, keys=prediction_df.keys())
    combined_df.columns = combined_df.columns.map('_'.join)
    combined_df.to_csv(f"{dirPath}prediction_{i}.csv", index=False)

def savePrediction(model, scaler, testDataList, dirName, isDebug=False):
    dirPath = f'../LSTM/Predictions/{dirName}/'
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(savePredictionCore, model, scaler, testGame, i, dirPath, isDebug)
            for i, testGame in enumerate(testDataList)
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting test games"):
            future.result()

def loadPrediction(testDataList, dirName):
    dirPath = f'../LSTM/Predictions/{dirName}/'
    predictions = []
    
    for i in range(len(testDataList)):
        file_path = f"{dirPath}prediction_{i}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            prediction_df = {col.split('_')[0]: df.filter(regex=f"^{col.split('_')[0]}_") for col in df.columns}
            for key in prediction_df:
                prediction_df[key].columns = [col.replace(f"{key}_", "") for col in prediction_df[key].columns]
            predictions.append(prediction_df)
        else:
            print(f"Warning: File {file_path} does not exist.")
            predictions.append(None)
    
    return predictions