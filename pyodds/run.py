import numpy as np
import pandas as pd
import argparse
import yaml
from pyodds.utils.importAlgorithm import algorithm_selection
from pyodds.utils.plotUtils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution
from pyodds.utils.utilities import str2bool


channel = 'temperature'

parser = argparse.ArgumentParser(description="Anomaly Detection Platform Settings")
args = parser.parse_args()

# Load the YAML file
with open(r'C:\Users\Thaibite Zeng\VT_research\GraphNN\pyodds\config.yaml', 'r') as f:
    config = yaml.safe_load(f)

for key, value in config.items():
    setattr(args, key, value)



#random seed setting
rng = np.random.RandomState(args.random_seed)
np.random.seed(args.random_seed)

clf = algorithm_selection(args.algorithm, random_state=rng, contamination=args.contamination)

#read training and data
data_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)

x_train = np.load(data_path + '/numpy/X_train_p.npy')
y_train = np.load(data_path + '/numpy/y_train_p.npy')

x_test = np.load(data_path + '/numpy/X_test.npy')
y_test = np.load(data_path + '/numpy/y_test.npy')

clf.fit(pd.DataFrame(x_train))

prediction_result = clf.predict(pd.DataFrame(x_test))
outlierness = clf.decision_function(pd.DataFrame(x_test))
anomaly_scores = clf.anomaly_likelihood(pd.DataFrame(x_test))


prediction_result[prediction_result == 1] = 0
prediction_result[prediction_result == -1] = 1

save_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\results\{}'.format(channel)
np.save(save_path + '/y_pred_LSTM', prediction_result)
