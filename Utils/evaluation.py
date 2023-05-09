# compare all the results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
import seaborn as sns


def plot_attention(attention_matrix, moteid, path):

    # Create the heatmap
    sns.set(font_scale=0.8)  # Set the font scale for better readability
    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(attention_matrix, annot=False, cmap='viridis', xticklabels=moteid, yticklabels=moteid)

    # Set the x-axis and y-axis labels
    plt.xlabel('Moteid')
    plt.ylabel('Moteid')

    plt.savefig(path + '/attention_score.png')
    # Show the plot
    plt.show()



def embedding_tsne(embedding, path):





    return


def hit_rate(test_scores, mote_mtrx):

    hit_rate_all = []
    for i in range(test_scores.shape[0]):
        error_motes = np.where(mote_mtrx[i] == 1)[0]
        num = error_motes.shape[0]
        if num > 0:
            pred_motes = np.argsort(test_scores[i])[::-1][:num]

            count = 0
            for id in pred_motes:
                if id in list(error_motes):
                    count +=1

            hit_rate = count/num
            hit_rate_all.append(hit_rate)
    return np.mean(hit_rate_all), np.max(hit_rate_all)




#%% Get Graph Results - Check the probelm
if __name__ == "__main__":

    channel = 'temperature'

    mote_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)
    abnormal_index = np.load(mote_path + '/testing_error_index.npy')
    error_mote_humid = np.load(mote_path + '/testing_error_mote_{}.npy'.format(channel))

    read_path  = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\results\{}'.format(channel)
    # GNN
    test_scores = np.load(read_path + '/test_scores.npy')
    # LSTM
    y_pred_LSTM = np.load(read_path + '/y_pred_LSTM.npy')

    test_labels = np.load(read_path + '/test_labels.npy')

    moteids_to_remove = [5, 15, 18, 28, 55, 56, 58]

    test_scores = test_scores.T
    mote_mtrx = np.zeros_like(test_scores)

    for ii, index in enumerate(abnormal_index):
        if index not in moteids_to_remove:
            index = index[index < mote_mtrx.shape[0]]
            # print(len(index))
            if len(index) > 0:
                mote_mtrx[index, error_mote_humid[ii]] = 1

    plt.imshow(mote_mtrx, aspect='auto', cmap='viridis')
    plt.show()

    plt.imshow(test_scores, aspect='auto', cmap='viridis')
    plt.show()

    print('hit rate: ', hit_rate(test_scores, mote_mtrx))

    # fig, ax = plt.subplots(1, 2, sharey=True)

    #%% LSTM results
    y_pred_LSTM = y_pred_LSTM[10:]
    f1, acc, recall = f1_score(y_pred_LSTM, test_labels), accuracy_score(y_pred_LSTM, test_labels), recall_score(y_pred_LSTM, test_labels)
    print(f1, acc, recall)
