from __future__ import division
from __future__ import print_function

import time


import numpy as np
import torch
import torch.nn as nn

from config import parser
from models.base_models import NCModel
from utils.data_utils import load_data



from layers.readout import AvgReadout
from layers.discriminator import Discriminator
from sklearn.metrics import  f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import os
import models.aug as aug


import warnings
warnings.filterwarnings('ignore')


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

# import args
args = parser.parse_args()
# load data
data = load_data(args, args.datapath)
args.n_nodes, args.feat_dim = data['features'].shape
args.n_classes = int(data['labels'].max() + 1)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'


Model = NCModel
model = Model(args)


labels = data['labels']



if not args.lr_reduce_freq:
    args.lr_reduce_freq = args.epochs

idx_train = torch.LongTensor(data['idx_train'])
idx_val = torch.LongTensor(data['idx_val'])
idx_test = torch.LongTensor(data['idx_test'])

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
read = AvgReadout()
sigm = nn.Sigmoid()
disc = Discriminator(args.dim)
knn = KNeighborsClassifier(n_neighbors=args.n_classes)

cnt_wait = 0
best = 1e9
best_t = 0
batch_size = 1

if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

print('------------Training------------------')
print('Datasets:', args.dataset)
print('Number of nodes:', args.n_nodes)
print('Number of classes:', args.n_classes)
print(str(model))
t_total = time.time()

if args.aug_method == 'Feature_drop':
    features_da =aug.drop_feature(data['features'], args.drop_rate)
    adj_da = data['adj_train_norm']
if args.aug_method == 'Random_mask':
    features_da = aug.aug_random_mask(data['features'], args.drop_rate)
    adj_da = data['adj_train_norm']

    
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    

    if args.aug_method == 'Corruption':
        idx = np.random.permutation(args.n_nodes)
        features_da = data['features'][idx, :]
        adj_da = data['adj_train_norm']


    lbl_1 = torch.ones(batch_size, args.n_nodes)
    lbl_2 = torch.zeros(batch_size, args.n_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)



    embeddings1 = model.encode(data['features'], data['adj_train_norm'])
    embeddings2 = model.encode(features_da, adj_da)
    if not args.cuda == -1:
        lbl = lbl.to(args.device)
        embeddings1 = embeddings1.to(args.device)
        embeddings2 = embeddings2.to(args.device)
    if args.manifold != 'Euclidean':
        e_embeddings1 = model.hyperbolic_to_euclidean(embeddings1)
        e_embeddings2 = model.hyperbolic_to_euclidean(embeddings2)
        if not args.cuda == -1:
            h_1 = torch.cuda.FloatTensor(e_embeddings1[np.newaxis])
            h_2 = torch.cuda.FloatTensor(e_embeddings2[np.newaxis])
        else:
            h_1 = torch.FloatTensor(e_embeddings1[np.newaxis])
            h_2 = torch.FloatTensor(e_embeddings2[np.newaxis])


    else:
        if not args.cuda == -1:
            h_1 = torch.cuda.FloatTensor(embeddings1[np.newaxis])
            h_2 = torch.cuda.FloatTensor(embeddings2[np.newaxis])
        else:
            h_1 = torch.FloatTensor(embeddings1[np.newaxis])
            h_2 = torch.FloatTensor(embeddings2[np.newaxis])
        
    z1 = model.decode(embeddings1, data['adj_train_norm'])
    z2 = model.decode(embeddings2, data['adj_train_norm'])
    
    c = read(h_1, None)
    c = sigm(c)
    if not args.cuda == -1:
        c = c.to(args.device)
        disc.to(args.device)
    ret = disc(c, h_1, h_2, None, None)
    loss = b_xent(ret, lbl)




    print('Epoch: {}   Loss: {:.4}   Time: {:.4}s'.format(epoch, loss, (time.time() - t)))

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == args.patience:
        print('Early stopping!')
        #break

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

print('Loading {}th epoch'.format(best_t))
print('Loading {}th epoch'.format(best_t))
print('Total time for training: {:.4}s'.format((time.time() - t_total)))
model.load_state_dict(torch.load('best_dgi.pkl'))

print('-----------------Testing-------------------')

embeds = model.encode(data['features'], data['adj_train_norm'])
e_embeds = model.hyperbolic_to_euclidean(embeds)
#embeds = torch.FloatTensor(embeds[np.newaxis])


X = e_embeds.detach().numpy()
Y = labels.detach().numpy()
Y = Y.reshape(-1, 1)
onehot_encoder = OneHotEncoder(categories='auto').fit(Y)

Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
X = normalize(X, norm='l2')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - 0.1)

logreg = LogisticRegression(solver='liblinear')

c = 2.0 ** np.arange(-10, 10)

clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_pred = prob_to_one_hot(y_pred)


micro = f1_score(y_test, y_pred, average="micro")
macro = f1_score(y_test, y_pred, average="macro")


print('micro:', micro)
print('macro:', macro)

