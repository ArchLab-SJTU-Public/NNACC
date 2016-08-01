from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
import math
import numpy as np
import json
import sys
 
def format_data(data):
    try:
        return data.reshape((data.shape[0], 1)) if len(data.shape) == 1 else data
    except AttributeError as e:
        print 'ERROR! data is not a numpy object, format_data failed!'
        exit(0)

def load_data(filename):
    data = np.loadtxt(filename)
    return format_data(data)

def build_model_ex(input_dim, output_dim, layers_dim):
    model = Sequential()
    model.add(Dense(layers_dim[0], input_dim=input_dim, init='uniform'))
    model.add(Activation('sigmoid'))

    for dim in layers_dim[1:]:
        model.add(Dense(dim, init='uniform'))
        model.add(Activation('sigmoid'))

    model.add(Dense(output_dim, init='uniform'))
    model.add(Activation('sigmoid'))

    prop = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=prop)
    return model

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(8, init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(output_dim, init='uniform'))
    model.add(Activation('sigmoid'))

    prop = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=prop)
    return model

def train_model(model, X, Y):
    A.fit(X, Y, nb_epoch=50000, batch_size=128, verbose=False)
    return A, A.evaluate(X, Y)


def train_model_seletion():
    # TODO
    pass



def main(xfilename, yfilename, outfile):
    X = load_data(xfilename)
    Y = load_data(yfilename)
    A = build_model(X.shape[1], Y.shape[1])
    A.fit(X, Y, nb_epoch=10, batch_size=128, verbose=True)
    with open(outfile, 'w') as f:
        layers = [l for l in A.layers if len(l.get_weights()) > 0]
        f.write(str(len(layers)) + '\n');
        for layer in layers:
            f.write(str(layer.get_weights()[0].shape[0]) + ' ');
        f.write(str(Y.shape[1]));
        f.write('\n')
        for layer in layers:
            weights = layer.get_weights()
            w = weights[0].shape[1]
            h = weights[0].shape[0]
            for i in range(h):
                s = ' '.join([str(v) for v in weights[0][i]]) + '\n'
                f.write(s)
            f.write(' '.join([str(v) for v in weights[1]]) + '\n')
                





if __name__ == "__main__":
    if len(sys.argv) != 4:
        print 'Usage: python train.py [input.data] [output.data] [result.nn]'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])



