import pandas as pd
import random

def printColumnsHasNan(df):
    print('Columns with NaN values:')
    for col in df.columns:
        if df[col].isnull().values.any():
            print(f'{col}')

def printNonNumericColumns(df):
    print('Non-numeric columns:')
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f'{col}')

def impute(df, exclude, idk, reference):
    df_process = df.copy()
    
    def removeColumns(columns):
        drops = [col for col in columns if col in df_process.columns]
        df_process.drop(drops, axis=1, inplace=True)
    
    def imputeColumn(column, value):
        if column in df_process.columns:
            df_process[column].fillna(value, inplace=True)
    
    def addPlayResult_removePlayType2():
        if 'playType2' in df_process.columns:
            playType2 = df_process['playType2'].str.split(' ', n=1, expand=True)
            df_process['playResult'] = playType2[1]
            df_process.drop('playType2', axis=1, inplace=True)
    
    removeColumns(exclude)
    removeColumns(idk)
    removeColumns(reference)
    addPlayResult_removePlayType2()
    
    imputeColumn('playType', -1)
    imputeColumn('playResult', -1)
    imputeColumn('distanceToGoalPre', -1)
    imputeColumn('fieldGoalProbability', -1)
    imputeColumn('huddle', 'misc')
    imputeColumn('formation', 'standard')

    return df_process

def convertGameClockToSeconds(df):
    df['gameClock'] = df['gameClock'].str.strip('()')  # Remove parentheses
    df['gameClock'] = df['gameClock'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    return df

def numericalize(df):
    df_process = df.copy()
    conversionTables = {}

    def getNonNumericColumns():
        return [col for col in df_process.columns if df_process[col].dtype == 'object']
    
    def numericalizeColumn(column):
        df_process[column], unique = pd.factorize(df_process[column])
        conversionTable = {category: code for code, category in enumerate(unique)}
        return conversionTable
    
    nonNumericColumns = getNonNumericColumns()
    for col in nonNumericColumns:
        conversionTables[col] = numericalizeColumn(col)
    
    return df_process, conversionTables

def makeConversionTablesIntoFile(conversionTables):
    for col, table in conversionTables.items():
        with open(f'../ConversionTables/{col}.txt', 'w') as file:
            for category, code in table.items():
                file.write(f'{category} {code}\n')

def runPreprocess(df, exclude, idk, reference):
    df_process = df.copy()
    df_process = impute(df_process, exclude, idk, reference)
    df_process = convertGameClockToSeconds(df_process)
    df_process, conversionTables = numericalize(df_process)
    makeConversionTablesIntoFile(conversionTables)
    return df_process

def getStringValue(feature, value):
    conversion_table = {}
    dirPath = '../ConversionTables/'
    file_path = f'{dirPath}{feature}.txt'
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                category = ' '.join(parts[:-1])
                code = parts[-1]
                conversion_table[int(code)] = category
    return conversion_table.get(value, "Unknown")


indices = ['playId', 'gameId']

playCircumstances = ['playSequence', 
                    'quarter', 
                    'possessionTeamId',
                    'nonpossessionTeamId', 
                    'playNumberByTeam',
                    'gameClock', 
                    'down', 
                    'distance',
                    'distanceToGoalPre',
                    'netYards',
                    'scorePossession',
                    'scoreNonpossession',
                    'fieldGoalProbability',]

playTypes = ['playType',
            'huddle',
            'formation']

playResults = ['playResult', # the second item of playType2
                'gameClockSecondsExpired',
              'gameClockStoppedAfterPlay', 
               'noPlay', # is the play a penalty
               'offensiveYards']

playSubsequences = ['isClockRunning', 
                        'changePossession', 
                        'turnover',
                        'safety',
                        'firstDown',]

idk = [ 'typeOfPlay',
        'fourthDownConversion',
        'thirdDownConversion',
        'homeScorePre', 
        'visitingScorePre',
        'homeScorePost',
        'visitingScorePost',
        'distanceToGoalPost']

# the original dataset has 3 columns of their own prediction of the play we may be able to use them as a reference
reference = ['evPre',
             'evPost', 
             'evPlay',]

exclude = [ 'playTypeDetailed', # redundant to playType2
            'fieldPosition', 
            'playDescription',
            'playStats',
            'playDescriptionFull', 
            'efficientPlay']

def getColumns(key):
    if key == 'playCircumstance':
        return playCircumstances
    elif key == 'playType':
        return playTypes
    elif key == 'playResult':
        return playResults
    elif key == 'playSubsequence':
        return playSubsequences
    elif key == 'idk':
        return idk
    elif key == 'reference':
        return reference
    elif key == 'exclude':
        return exclude
    elif key == 'input':
        return [item for item in playCircumstances if item != 'gameClock'] + playSubsequences
    else:
        return []

def getCircumstance(df):
    return df[playCircumstances]
def getPlayType(df):
    return df[playTypes]
def getPlayResult(df):
    return df[playResults]
def getInput(df):
    return df[getColumns('input')]

def getSplittedList(df):
    # split_list = [[game0], [game1], [game2], ...]
    split_list = [df[df['gameId'] == value] for value in df['gameId'].unique()]
    return split_list

def separateTrainTest(splitList, test_size=0.2):
    Ntotal = len(splitList)
    Ntest = int(Ntotal * test_size)
    
    i_test = random.sample(range(Ntotal), Ntest)
    
    trainList = [splitList[i] for i in range(Ntotal) if i not in i_test]
    testList = [splitList[i] for i in i_test]
    
    return trainList, testList
     