import pandas as pd
import numpy as np
import datetime
import random
pd.set_option('display.max_columns', 500)

np.random.seed(531)


def identify_attacks(test_data):
    """
    
    Given the test_data identifies the attack intervals and creates a pandas DataFrame where those spoofing is going to be applied.
    
    Returns
    -------
    DataFrame
        summary of the attack intervals
    """
    # find attacks among data
    attacks = test_data.loc[test_data['Normal/Attack'] == 'Attack']
    prev_datetime = attacks.index[0]  # find first timing
    start = prev_datetime
    count_attacks = 0

    days_in_advance = 5.5
    # find attacks bounds
    attack_intervals = pd.DataFrame(
        columns=['Name', 'Start', 'End', 'Replay_Copy'])
    for index, _ in attacks.iterrows():
        # datetime.timedelta(minutes=15)): #change attack
        if (index - prev_datetime > datetime.timedelta(seconds=1)):
            count_attacks = count_attacks + 1
            interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, (start - datetime.timedelta(days=days_in_advance))]], 
                                     columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
            attack_intervals = attack_intervals.append(interval)
            start = index
        prev_datetime = index
    count_attacks = count_attacks + 1
    interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, (start - datetime.timedelta(days=days_in_advance))]], 
                                     columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
    attack_intervals = attack_intervals.append(interval)
    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals


def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, constraints=None):
    
    """
    
    given a spoofing_technique to be applied, the attack_intervals, eavesdropped_data and test_data, it builds the dataset containing sensor spoofing.
    
    Returns
    -------
    DataFrame
        Dataset with spoofed sensor readings.
    """
    prev_end = test_data.index[0]
    df2 = pd.DataFrame()
    # replay data
    for index, row in attack_intervals.iterrows():
        if index == 1:
            df2 = df2.append(test_data.loc[prev_end: row['Start']-datetime.timedelta(seconds=1)])
        else:
            df2 = df2.append(test_data.loc[prev_end+datetime.timedelta(
                seconds=1): row['Start']-datetime.timedelta(seconds=1)])
        df = pd.DataFrame(columns=eavesdropped_data.columns)
        if constraints:
            print(constraints[index-1])
            df = spoofing_technique(df,
                                    row, eavesdropped_data, test_data, attack_intervals, constraints[index-1])
        else:
            df = spoofing_technique(df,
                                    row, eavesdropped_data, test_data, attack_intervals)
        df['Normal/Attack'] = 'Attack'
        df3 = pd.DataFrame(data=df.values, columns=df.columns,
                           index=test_data.loc[row['Start']: row['End']].index)  # update datetime
        df2 = df2.append(df3, ignore_index=False)[df3.columns.tolist()]
        prev_end = row['End']
    df2 = df2.append(
        test_data.loc[prev_end+datetime.timedelta(seconds=1): test_data.last_valid_index()])
    return df2


def replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    applies replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df['Normal/Attack'] = ''
    df = df.append(eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy']+(
        row['End']-(row['Start']))])[test_data.columns.tolist()]  # append replayed row
    return df


def random_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    applies random replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df = df.append(eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy']+(
        row['End']-(row['Start']))].sample(frac=1, random_state = 531))[test_data.columns.tolist()]  # append replayed row
    return df


def constant(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    
    applies constant attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied constant attack
    """
    mean = eavesdropped_data.mean(axis=0)
    length = len(test_data.loc[row['Start']:row['End']])
    for column in eavesdropped_data:
        constant = [mean[column]]*length
        if column in actuator_columns:
            constant = [round(x) for x in constant]
        df[column] = constant
    return df


def stale(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    length = len(test_data.loc[row['Start']:row['End']])
    stale=df.append(test_data.loc[row['Start']-datetime.timedelta(seconds=1)])[test_data.columns.tolist()]
    df = pd.concat([stale]*length)
    return df


def gaussian_noise(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    applies gaussian noise attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied gaussian noise attack
    """
    length = len(test_data.loc[row['Start']:row['End']])
    mean = eavesdropped_data.mean(axis=0)
    std = eavesdropped_data.std(axis=0)
    for column in df:
        noise = np.random.normal(mean[column], std[column], length)
        if column in actuator_columns:
            noise = noise.round()
            noise[noise > 2] = 2
            noise[noise < 0] = 0
            df[column] = noise.astype(int)
        else:
            df[column] = noise
    return df

def gaussian_noise_2(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    applies gaussian noise attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied gaussian noise attack
    """
    length = len(test_data.loc[row['Start']:row['End']])
    mean = eavesdropped_data.mean(axis=0)
    std = eavesdropped_data.std(axis=0)
    for column in df: 
        noise = np.random.normal(mean[column], std[column], length)
        if column in actuator_columns:
            noise = noise.round()
            noise = noise + 5
            df[column] = noise.astype(int)
        else:
            df[column] = noise
    return df


def constrained_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    constrained version of the replay attack
    """
    
    constraints = args[0]
    check_constraints(constraints)
    df = df.append(test_data.loc[row['Start']: row['End']])
    df[constraints] = eavesdropped_data[constraints].loc[row['Replay_Copy']
        :row['Replay_Copy']+(row['End']-(row['Start']))].values
    df[actuator_columns] = df[actuator_columns].astype(int)
    return df


def constrained_random_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    constrained version of the replay attack
    """

    constraints = args[0]
    check_constraints(constraints)
    df = df.append(test_data.loc[row['Start']: row['End']])
    df[constraints] = eavesdropped_data[constraints].loc[row['Replay_Copy']
        :row['Replay_Copy']+(row['End']-(row['Start']))].sample(frac=1, random_state = 531).values
    return df


def constrained_constant(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    constrained version of the constant attack
    """
    
    constraints = args[0]
    check_constraints(constraints)
    mean = eavesdropped_data.mean(axis=0)
    df = df.append(test_data.loc[row['Start']: row['End']])
    length = len(test_data.loc[row['Start']:row['End']])
    for column in constraints:
        constant = [mean[column]]*length
        if column in actuator_columns:
            constant = [round(x) for x in constant]
        df[column] = constant
    return df


def constrained_gaussian(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    constrained version of the gaussian noise attack
    """
    constraints = args[0]
    check_constraints(constraints)
    length = len(test_data.loc[row['Start']:row['End']])
    mean = eavesdropped_data.mean(axis=0)
    std = eavesdropped_data.std(axis=0)
    df = df.append(test_data.loc[row['Start']: row['End']])
    for column in constraints:
        noise = np.random.normal(mean[column], std[column], length)
        if column in actuator_columns:
            noise = noise.round()
            noise[noise > 2] = 2
            noise[noise < 0] = 0
            df[column] = noise.astype(int)
        else:
            df[column] = noise
    return df

def constrained_gaussian_2(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    constrained version of the gaussian_noise_2 attack
    """
    constraints = args[0]
    check_constraints(constraints)
    length = len(test_data.loc[row['Start']:row['End']])
    mean = eavesdropped_data.mean(axis=0)
    std = eavesdropped_data.std(axis=0)
    df = df.append(test_data.loc[row['Start']: row['End']])
    for column in constraints:
        noise = np.random.normal(mean[column], std[column], length)
        if column in actuator_columns:
            noise = noise.round()
            noise = noise + 5
            df[column] = noise.astype(int)
        else:
            df[column] = noise
    return df

def constrained_stale(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    constraints = args[0]
    check_constraints(constraints)
    length = len(test_data.loc[row['Start']:row['End']])
    stale=df.append(test_data.loc[row['Start']-datetime.timedelta(seconds=1)])[test_data.columns.tolist()]
    #print(stale)
    df = df.append(test_data.loc[row['Start']: row['End']])
    for column in constraints:
        value = stale[column].values
        df[column] = [value[0]]*length
    return df

def check_constraints(constraints):
    if constraints == None:
        print('Provide constraints')
        import sys
        sys.exit()
    else:
        pass




if __name__ == "__main__":
    
    """
    Material for anonymous submission
    The spoofing framework requires:
    test_data : data organized in a .csv file where the the first row represent the features names, and all the others contains the sampled data at every time step.
    eavesdropped_data : data used to train the spoofing technique.
    
    in case of constrained attacks you need the files containing the constraints to be applied.
    
    then the script saves the dataset with applied spoofing into the file system

    """
    test_data = pd.read_csv('./SWAT/SWaT_Dataset_Attack_v1.csv',
                            index_col=['Timestamp'], parse_dates=True, dayfirst=True)
    eavesdropped_data = pd.read_csv("./SWAT/SWaT_Dataset_Normal_v1.csv", index_col=[
                                    'Timestamp'], parse_dates=True,  dayfirst=True)
    
    
    unconstrained_spoofing_techniques = [replay, stale]#gaussian_noise, gaussian_noise_2, stale, replay, constant]
    constrained_spoofing_techniques = []#constrained_stale, constrained_replay, constrained_constant, constrained_gaussian, constrained_gaussian_2]
    actuator_columns = test_data.filter(regex=("(MV|P[0-9]|UV)")).columns.tolist()
    attack_intervals = identify_attacks(test_data)
    
    if unconstrained_spoofing_techniques:
        
        for spoofing_technique in unconstrained_spoofing_techniques:
            print('_________________')
            print(spoofing_technique)
            print('_________________')
            spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data)
            spoofed_data.to_csv('./SWAT/unconstrained_spoofing/'+spoofing_technique.__name__+'.csv')
            
    if constrained_spoofing_techniques:
        for i in [1]:
            constraints=[]
            for att_num in range(1,35):
                s = open('./SWAT/constraints/constraints_attack'+str(att_num)+'.txt', 'r').read()
                dictionary =  eval(s)
                try:
                    constraints.append(dictionary[i])
                except:
                    constraints.append(dictionary[i+1])
            
            print(constraints)

            for spoofing_technique in constrained_spoofing_techniques:
                print('_________________')
                print(spoofing_technique)
                print('_________________')
                spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data, constraints)
                spoofed_data.to_csv('./SWAT/constrained_spoofing/'+spoofing_technique.__name__+'_allowed_'+str(i)+'.csv',  float_format='%.6f')
