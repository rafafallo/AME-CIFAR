# Copyright [2021] Luis Alberto Pineda Cortés,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import numpy as np
from matplotlib import pyplot as plt

# Keys for data
LOSS = 'loss'
C_LOSS = 'classification_loss'
A_LOSS = 'autoencoder_loss'
C_ACCURACY = 'classification_accuracy'
A_ACCURACY = 'autoencoder_accuracy'
VAL = 'val_'


def trplot(a_measure, b_measure, a_label, b_label, max_epoch, nn):
    epoch = len(a_measure)
    epoch = max_epoch if epoch > max_epoch else epoch

    fig = plt.figure()
    x = np.arange(0,epoch)
    plt.errorbar(x, a_measure[:epoch], fmt='b-.', label=a_label)
    plt.errorbar(x, b_measure[:epoch], fmt='r--,', label=b_label)
    plt.legend(loc=0)

    plt.suptitle(f'Neural net No. {nn}')
    plt.show()
    

def teplot(a_measure, b_measure, a_label, b_label):
    fig = plt.figure()
    x = np.arange(0,len(a_measure))
    plt.errorbar(x, a_measure, fmt='b-.', label=a_label)
    plt.errorbar(x, b_measure, fmt='r--,', label=b_label)
    plt.legend(loc=0)

    plt.suptitle(f'Average results')
    plt.show()


def training_stats(data, max_epoch):
    """ Analyse neural nets training data. 
        
        Training stats data is a list of dictionaries with the full
        set of keys declared above.
    """
    n = 0
    for d in data:
        trplot(d[LOSS], d[VAL+LOSS], LOSS, VAL+LOSS,max_epoch,n)
        trplot(d[C_LOSS], d[VAL+C_LOSS], C_LOSS, VAL+C_LOSS,max_epoch,n)
        trplot(d[A_LOSS], d[VAL+A_LOSS], A_LOSS, VAL+A_LOSS,max_epoch,n)
        trplot(d[C_ACCURACY], d[VAL+C_ACCURACY], C_ACCURACY, VAL+C_ACCURACY,max_epoch,n)
        trplot(d[A_ACCURACY], d[VAL+A_ACCURACY], A_ACCURACY, VAL+A_ACCURACY,max_epoch,n)
        n += 1


def testing_stats(data):
    """ Analyse neural nets testing data. 
    """

    n = len(data)
    m = {LOSS: [],
         C_LOSS: [],
         A_LOSS: [],
         C_ACCURACY: [],
         A_ACCURACY: []}
    
    for d in data:
        m[LOSS].append(d[LOSS])
        m[C_LOSS].append(d[C_LOSS])
        m[A_LOSS].append(d[A_LOSS])
        m[C_ACCURACY].append(d[C_ACCURACY])
        m[A_ACCURACY].append(d[A_ACCURACY])

    teplot(m[C_LOSS],m[A_LOSS],C_LOSS,A_LOSS)
    teplot(m[C_ACCURACY],m[A_ACCURACY],C_ACCURACY,A_ACCURACY)


if __name__== "__main__" :
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} file.json epochs.')
        sys.exit(1)

    fname = sys.argv[1]
    max_epoch = int(sys.argv[2])
    
    history = {}
    with open(fname) as json_file:
        history = json.load(json_file)
    history = history['history']

    # Now, history contains a list with the statistics from the neural nets.
    # Odd elements have statistics from training and validation, while
    # even elements have statistics from testing.

    training = []
    testing = []

    odd = True
    for s in history:
        if odd:
            training.append(s)
        else:
            testing.append(s)
        odd = not odd

    testing_stats(testing)
    training_stats(training, max_epoch)
