from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import keras
from keras import activations
import argparse
import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auc_score
import keras
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, GRU, Lambda, Permute
from keras.models import Model

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=4,
                help="# of GPUs to use for training")
ap.add_argument("-batch", "--batch_size", type=int, default=256,
                help="# batch size to use for training")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs for training")
ap.add_argument("-ref", "--reference_points", type=int,
                default=192, help="# of reference points")
ap.add_argument("-units", "--hidden_units", type=int,
                default=100, help="# of hidden units")
ap.add_argument("-hfadm", "--hours_from_adm", type=int,
                default=48, help="Hours of record to look at")
ap.add_argument('--task', type=str, default='in_hospital_mortality')

args = vars(ap.parse_args())
gpu_num = args["gpus"]
epoch = args["epochs"]
hid = args["hidden_units"]
ref_points = args["reference_points"]
hours_look_ahead = args["hours_from_adm"]
if gpu_num > 0:
    batch = args["batch_size"]*gpu_num
else:
    batch = args["batch_size"]



class single_channel_interp(Layer):

    def __init__(self, ref_points, hours_look_ahead, **kwargs):
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead  # in hours
        super(single_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape [batch, features, time_stamp]
        self.time_stamp = input_shape[2]
        self.d_dim = input_shape[1] // 4
        self.activation = activations.get('sigmoid')
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.d_dim, ),
            initializer=keras.initializers.Constant(value=0.0),
            trainable=True)
        super(single_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        d = x[:, 2*self.d_dim:3*self.d_dim, :]
        if reconstruction:
            output_dim = self.time_stamp
            m = x[:, 3*self.d_dim:, :]
            ref_t = K.tile(d[:, :, None, :], (1, 1, output_dim, 1))
        else:
            m = x[:, self.d_dim: 2*self.d_dim, :]
            ref_t = np.linspace(0, self.hours_look_ahead, self.ref_points)
            output_dim = self.ref_points
            ref_t.shape = (1, ref_t.shape[0])
        #x_t = x_t*m
        d = K.tile(d[:, :, :, None], (1, 1, 1, output_dim))
        mask = K.tile(m[:, :, :, None], (1, 1, 1, output_dim))
        x_t = K.tile(x_t[:, :, :, None], (1, 1, 1, output_dim))
        norm = (d - ref_t)*(d - ref_t)
        a = K.ones((self.d_dim, self.time_stamp, output_dim))
        pos_kernel = K.log(1 + K.exp(self.kernel))
        alpha = a*pos_kernel[:, np.newaxis, np.newaxis]
        w = K.logsumexp(-alpha*norm + K.log(mask), axis=2)
        w1 = K.tile(w[:, :, None, :], (1, 1, self.time_stamp, 1))
        w1 = K.exp(-alpha*norm + K.log(mask) - w1)
        y = K.sum(w1*x_t, axis=2)
        if reconstruction:
            rep1 = tf.concat([y, w], 1)
        else:
            w_t = K.logsumexp(-10.0*alpha*norm + K.log(mask),
                              axis=2)  # kappa = 10
            w_t = K.tile(w_t[:, :, None, :], (1, 1, self.time_stamp, 1))
            w_t = K.exp(-10.0*alpha*norm + K.log(mask) - w_t)
            y_trans = K.sum(w_t*x_t, axis=2)
            rep1 = tf.concat([y, w, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], 2*self.d_dim, self.time_stamp)
        return (input_shape[0], 3*self.d_dim, self.ref_points)


class cross_channel_interp(Layer):

    def __init__(self, **kwargs):
        super(cross_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_dim = input_shape[1] // 3
        self.activation = activations.get('sigmoid')
        self.cross_channel_interp = self.add_weight(
            name='cross_channel_interp',
            shape=(self.d_dim, self.d_dim),
            initializer=keras.initializers.Identity(gain=1.0),
            trainable=True)

        super(cross_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = K.int_shape(x)[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = K.exp(w)
        y = tf.transpose(y, perm=[0, 2, 1])
        w = tf.transpose(w, perm=[0, 2, 1])
        w2 = w
        w = K.tile(w[:, :, :, None], (1, 1, 1, self.d_dim))
        den = K.logsumexp(w, axis=2)
        w = K.exp(w2 - den)
        mean = K.mean(y, axis=1)
        mean = K.tile(mean[:, None, :], (1, self.output_dim, 1))
        w2 = K.dot(w*(y - mean), cross_channel_interp) + mean
        rep1 = tf.transpose(w2, perm=[0, 2, 1])
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = tf.concat([rep1, intensity, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, self.output_dim)
        return (input_shape[0], 3*self.d_dim, self.output_dim)


def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


def mean_imputation(vitals, mask):
    """For the time series missing entirely, our interpolation network
    assigns the starting point (time t=0) value of the time series to
    the global mean before applying the two-layer interpolation network.
    In such cases, the first interpolation layer just outputs the global
    mean for that channel, but the second interpolation layer performs
    a more meaningful interpolation using the learned correlations from
    other channels."""
    counts = np.sum(np.sum(mask, axis=2), axis=0)
    mean_values = np.sum(np.sum(vitals*mask, axis=2), axis=0)/counts
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i, j]) == 0:
                mask[i, j, 0] = 1
                vitals[i, j, 0] = mean_values[j]
    return

def customloss(ytrue, ypred):
    """ Autoencoder loss
    """
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
                   3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
    wc.shape = (1, num_features)
    y = ytrue[:, :num_features, :]
    m2 = ytrue[:, 3*num_features:4*num_features, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_features:2*num_features, :]
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    x = x/(wc**2)  # dividing by standard deviation
    x = tf.reduce_sum(x, axis=1)/num_features
    return tf.reduce_mean(x)

def interp_net():
    if gpu_num > 1:
        dev = "/cpu:0"
    else:
        dev = "/gpu:0"
    with tf.device(dev):
        main_input = Input(shape=(4*num_features, timestamp), name='input')
        sci = single_channel_interp(ref_points, hours_look_ahead)
        cci = cross_channel_interp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True),
                      reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh', recurrent_dropout=0.2, dropout=0.2)(z)
        main_output = Dense(1, activation='sigmoid', name='main_output')(z)
        orig_model = Model([main_input], [main_output, aux_output])
    if gpu_num > 1:
        model = multi_gpu_model(orig_model, gpus=gpu_num)
    else:
        model = orig_model
    print(orig_model.summary())
    return model

if __name__ == '__main__':

    x = np.load('../Dataset/'+args.task+'/input.npy', allow_pickle=True)
    y = np.load('../Dataset/'+args.task+'/output.npy', allow_pickle=True)

    print(x.shape, y.shape)
    timestamp = x.shape[2]
    num_features = x.shape[1] // 4


    seed = 0
    results = {}
    results['loss'] = []
    results['auc'] = []
    results['acc'] = []
    results['auprc'] = []

    # interpolation-prediction network

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0000, patience=20, verbose=0)
    callbacks_list = [earlystop]

    # 5-fold cross-validation

    i = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in kfold.split(np.zeros(len(y)), y):
        print("Running Fold:", i+1)
        model = interp_net()  # re-initializing every time
        model.compile(
            optimizer='adam',
            loss={'main_output': 'binary_crossentropy', 'aux_output': customloss},
            loss_weights={'main_output': 1., 'aux_output': 1.},
            metrics={'main_output': 'accuracy'})
        model.fit(
            {'input': x[train]}, {'main_output': y[train], 'aux_output': x[train]},
            batch_size=batch,
            callbacks=callbacks_list,
            nb_epoch=epoch,
            validation_split=0.20,
            verbose=2)
        y_pred = model.predict(x[test], batch_size=batch)
        y_pred = y_pred[0]
        total_loss, score, reconst_loss, acc = model.evaluate(
            {'input': x[test]},
            {'main_output': y[test], 'aux_output': x[test]},
            batch_size=batch,
            verbose=0)
        results['loss'].append(score)
        results['acc'].append(acc)
        results['auc'].append(auc_score(y[test], y_pred))
        results['auprc'].append(auprc(y[test], y_pred))
        print(results)
        i += 1