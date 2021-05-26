import numpy as np
from timeit import default_timer as timer

try:
    import cPickle as pickle
except:
    import pickle
from keras.models import load_model

from n3jet.models import Model

data_dir = "./test_data/"
model_dir = "./test_model/"

mom_file = data_dir + "3g2A_test_momenta.npy"
nj_file = data_dir + "3g2A_test_nj.npy"

try:
    test_momenta = np.load(mom_file, allow_pickle=True)
    test_nj = np.load(nj_file, allow_pickle=True)
except:
    test_momenta = np.load(mom_file, allow_pickle=True, encoding="latin1")
    test_nj = np.load(nj_file, allow_pickle=True, encoding="latin1")


test_momenta = test_momenta.tolist()

nlegs = len(test_momenta[0]) - 2

NN = Model(
    input_size=(nlegs + 2) * 4,
    momenta=test_momenta,
    labels=test_nj,
    all_jets=False,
    all_legs=True,
    high_precision=False,
)

_, _, _, _, _, _, _, _ = NN.process_training_data()

model = load_model(
    model_dir + "model",
    custom_objects={"root_mean_squared_error": NN.root_mean_squared_error},
)

pickle_out = open(model_dir + "dataset_metadata.pickle", "rb")
metadata = pickle.load(pickle_out)
pickle_out.close()

x_mean = metadata["x_mean"]
y_mean = metadata["y_mean"]
x_std = metadata["x_std"]
y_std = metadata["y_std"]

x_standard = NN.process_testing_data(
    moms=test_momenta, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std
)

start = timer()
pred = model.predict(x_standard)
end = timer()
elapsed = end - start

print(
    "Total time elapsed for evaluation of {} points = {}s".format(
        len(test_momenta), elapsed
    )
)
print("Evaluation time per point = {}s".format(elapsed / float(len(test_momenta))))
