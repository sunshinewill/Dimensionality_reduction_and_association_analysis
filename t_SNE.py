import argparse
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='Process some dataset.')
parser.add_argument('fileName', metavar='File Name', help='A file name for the program')
args = parser.parse_args()
fileName = args.fileName
# fileName = "pca_demo.txt"
df = pds.read_csv(
    filepath_or_buffer=fileName,
    header=None,
    sep="\t")
dim = df.shape[1]-1
num_data = df.shape[0]

# df.columns=['0', '1', '2', '3', 'Disease']
# print(df)

x = df.iloc[:, :dim].values
y = df.iloc[:, -1].values
# print(x)
# X_std = StandardScaler().fit_transform(x)

mean_vec = np.mean(x, axis=0)
x_norm = x - mean_vec

trans = TSNE(n_components=2).fit_transform(x)

# Plot
# Y = np.concatenate((pc, y.reshape(y.shape[0], 1)), axis=1)
hasharray = np.zeros([num_data, 1])

for i in range(num_data):
    hasharray[i, 0] = hash(y[i])

plt.scatter(trans[:, 0], trans[:, 1], c=hasharray[:, 0])
plt.title('Scatter plot by t_SNE on '+fileName.split('.')[0])
plt.xlabel('x')
plt.ylabel('y')
plt.show()