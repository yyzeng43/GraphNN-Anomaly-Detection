# from pyodds.algo.iforest import IFOREST
# from pyodds.algo.ocsvm import OCSVM
# from pyodds.algo.lof import LOF
# from pyodds.algo.robustcovariance import RCOV
# from pyodds.algo.staticautoencoder import StaticAutoEncoder
# from pyodds.algo.luminolFunc import luminolDet
# from pyodds.algo.cblof import CBLOF
# from pyodds.algo.knn import KNN
# from pyodds.algo.hbos import HBOS
# from pyodds.algo.sod import SOD
# from pyodds.algo.pca import PCA
# from pyodds.algo.dagmm import DAGMM
# from pyodds.algo.lstmad import LSTMAD
from pyodds.algo.lstmencdec import LSTMED


def algorithm_selection(algorithm,random_state,contamination):
    """
    Select algorithm from tokens.

    Parameters
    ----------
    algorithm: str, optional (default='iforest', choices=['iforest','lof','ocsvm','robustcovariance','staticautoencoder','luminol','cblof','knn','hbos','sod','pca','dagmm','autoencoder','lstm_ad','lstm_ed'])
        The name of the algorithm.
    random_state: np.random.RandomState
        The random state from the given random seeds.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    Returns
    -------
    alg: class
        The selected algorithm method.

    """
    algorithm_dic={
                   'lstm_ed':LSTMED(contamination=contamination,num_epochs=10, batch_size=20, lr=1e-3,hidden_size=5, sequence_length=30, train_gaussian_percentage=0.25)
                   }
    alg = algorithm_dic[algorithm]
    return alg
