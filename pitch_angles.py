import pandas as pd
import numpy as np

## Needed Columns
# [
#   'extension',
#   'vX0', # initial velo, X direction
#   'vY0', # initial velo, Y direction
#   'vZ0', # initial velo, Z direction
#   'aX',  # acceleration, X direction
#   'aY', # acceleration, Y direction
#   'aZ', # acceleration, Z direction
#   'pZ' # plate location, Z direction
# ]

def pitch_angles(dataframe):
    ### Physical characteristics of pitch
    ## Release Angles

    # Release Speed
    dataframe['vYs'] = -((dataframe['vY0']**2 - 2 * dataframe['aY'] * (60.5 - dataframe['extension'] - 50)) ** 0.5)

    # Time to plate, from start
    dataframe['pitch_time_start'] = (dataframe['vYs'] - dataframe['vY0'])/dataframe['aY']

    # Release speed, X- and Z-directions
    dataframe['vXs'] = dataframe['vX0'] - dataframe['aX'] * dataframe['pitch_time_start']
    dataframe['vZs'] = dataframe['vZ0'] - dataframe['aZ'] * dataframe['pitch_time_start']

    # Release Angles, Horizontal, and Vertical
    dataframe['HRA'] = -1 * np.arctan(dataframe['vXs']/dataframe['vYs']) * (180/np.pi)
    dataframe['VRA'] = -1 * np.arctan(dataframe['vZs']/dataframe['vYs']) * (180/np.pi)
    
    # Pitch velocity (to plate) at plate
    dataframe['vYf'] = -1 * (dataframe['vY0']**2 - (2 * dataframe['aY']*(50-17/12)))**0.5
    # Pitch time in air (50ft to home plate)
    dataframe['pitch_time_50ft'] = (dataframe['vYf'] - dataframe['vY0'])/dataframe['aY']
    # Pitch velocity (vertical) at plate
    dataframe['vXf'] = dataframe['vX0'] + dataframe['aX'] * dataframe['pitch_time_50ft']
    dataframe['vZf'] = dataframe['vZ0'] + dataframe['aZ'] * dataframe['pitch_time_50ft']
    
    ## raw HAA, raw VAA, and height-adjusted VAA
    # Raw HAA
    dataframe['HAA'] = -1 * np.arctan(dataframe['vXf']/dataframe['vYf']) * (180/np.pi)
    # Raw VAA
    dataframe['VAA'] = -1 * np.arctan(dataframe['vZf']/dataframe['vYf']) * (180/np.pi)
    # Adjust for VAA of pitches at that height
    dataframe['vaa_z_adj'] = np.where(dataframe['pZ']<3.5,
                                      dataframe['pZ'].mul(1.5635).add(-10.092),
                                      dataframe['pZ'].pow(2).mul(-0.1996).add(dataframe['pZ'].mul(2.704)).add(-11.69))
    # Adjusted VAA, based on height
    dataframe['HAVAA'] = dataframe['VAA'].sub(dataframe['vaa_z_adj'])

    return dataframe[['HRA','VRA','HAA','VAA','HAVAA']]
