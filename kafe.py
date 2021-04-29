"""
@author: awaash
"""


import argparse
import numpy as np
import pickle
import torch
import umap
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from data_reader import DataReader, KAFEdataset
from model import KAFEModel, AdvModel


parser = argparse.ArgumentParser(description='KAFE arguments')
parser.add_argument('--data', type=str, default='data_in/',
                    help='Location of the input data file. Txt format. One line per visit')
parser.add_argument('--emb_dimension', type=int, default=32,
                    help='Word embeddings size')
parser.add_argument('--initial_lr', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('--adv_bias', type=float, default=75.0,
                    help='Adversarial bias. Distinction between rare and frequent words')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size (No. of visits)')
parser.add_argument('--min_count', type=int, default=1,
                    help='Minimum count')
parser.add_argument('--window_size', type=int, default=25,
                    help='window size')
parser.add_argument('--neg_samples', type=int, default=5,
                    help='Negative samples')
parser.add_argument('--iterations', type=int, default=300,
                    help='Training iterations')
parser.add_argument('--root_size', type=int, default=3,
                    help='Root definition')
parser.add_argument('--param_lambda', type=float, default=0.5,
                    help='Lambda. Trade-off between skip-gram and discriminator loss')
parser.add_argument('--output_dim', type=int, default=1,
                    help='Discriminator output dimension')
args = parser.parse_args(args=[])
args.tied = True
print(args)

class KAFEtrainer:
  
    def __init__(self, input_file, args):
        self.adv_bias = args.adv_bias
        self.output_dim = args.output_dim
        self.data = DataReader(input_file, args.min_count, args.root_size, self.adv_bias)
        dataset = KAFEdataset(self.data, args.window_size, args.neg_samples)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=1, collate_fn=dataset.collate)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.emb_dimension
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.min_count = args.min_count
        self.KAFE_model = KAFEModel(self.emb_size, len(self.data.root2id), self.emb_dimension)
        self.weight_dict, self.root_dict, self.alpha_dict = {}, {}, {}
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.param_lambda = args.param_lambda
        self.adv_model = AdvModel(self.emb_dimension, self.output_dim)
        if self.use_cuda:
            self.KAFE_model.cuda()
            self.adv_model.cuda()
        print(min(self.data.discards), max(self.data.discards))

    def train(self):
        adv_losses_a, adv_losses_b, skipgram_losses, kafe_losses = [], [], [], []
        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer_dense = optim.Adam(self.KAFE_model.parameters())
            adv_criterion = nn.BCEWithLogitsLoss()
            adv_optimizer = optim.Adam(self.adv_model.parameters())
            adv_targets = torch.FloatTensor(list(self.data.word_label.values())).to(self.device)

            for i, sample_batched in enumerate(tqdm(self.dataloader,disable=True)):
                #sample_batched = next(iter(tqdm(self.dataloader)))
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    pos_r = sample_batched[3].to(self.device)

                    optimizer_dense.zero_grad()
                    adv_optimizer.zero_grad()
                    skipgram_loss = self.KAFE_model.forward(pos_u, pos_v, neg_v, pos_r)
                    g = (self.KAFE_model.u_embeddings.weight * self.KAFE_model.alpha.view(self.emb_size,1)) + \
                        (self.KAFE_model.r_embeddings.weight[self.data.root_map,:] * (1-self.KAFE_model.alpha).view(self.emb_size,1))
                    
                    adv_out = self.adv_model(g.detach())
                    adv_loss_a = adv_criterion(adv_out.squeeze(), adv_targets)
                    adv_loss_a.backward()
                    adv_optimizer.step()
                    optimizer_dense.zero_grad()
                    adv_optimizer.zero_grad()
                    x = (adv_targets == 1).nonzero(as_tuple=False)
                    adv_out = self.adv_model(g[x,:])                       
                    adv_loss_b = adv_criterion(adv_out.squeeze(), adv_targets[x.squeeze()])
                    loss = skipgram_loss - self.param_lambda*adv_loss_b
                    loss.backward()
                    optimizer_dense.step()

                    adv_losses_a.append(adv_loss_a.item())
                    adv_losses_b.append(adv_loss_b.item())
                    kafe_losses.append(loss.item())
                    skipgram_losses.append(skipgram_loss.item())


            print(str(iteration + 1)," Adv_Loss_A: {:.4f}, Adv_Loss_B: {:.4f}, Skipgram_Loss: {:.4f}, Total_loss: {:.4f}".
                  format(np.mean(adv_losses_a), np.mean(adv_losses_b), np.mean(skipgram_losses)
                         , np.mean(kafe_losses)))
            
            adv_losses_a, adv_losses_b, kafe_losses, skipgram_losses = [], [], [], []

        output = (self.data, self.KAFE_model)
        with open('KAFE_out.pkl', 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    def kafe_plot(self,n):
        g = ((self.KAFE_model.u_embeddings.weight * self.KAFE_model.alpha.view(self.emb_size,1)) + \
                        (self.KAFE_model.r_embeddings.weight[self.data.root_map,:] * (1-self.KAFE_model.alpha).view(self.emb_size,1))).detach().cpu()
        rare = [self.data.id2word[i] for i, j in self.data.word_label.items() if j ==1]
        frequent = [self.data.id2word[i] for i, j in self.data.word_label.items() if j == 0]
        labels_sub = random.sample(frequent,n) + random.sample(rare,n)
        random.shuffle(labels_sub)
        
        temp = [self.data.word2id[i] for i in labels_sub]
        g_sub = g[temp, :]
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, metric='cosine')
        Y = reducer.fit_transform(g_sub)
        
        freq_labels = [self.data.word_label[self.data.word2id[i]] for i in labels_sub]
        
        plt_clr = {}
        x = list(set([i[0] for i in self.data.word2id.keys()]))
        for i in range(0,len(x)):
            plt_clr[x[i]]=i

        ep = 0.5
        cmap = plt.cm.get_cmap('hsv', len(x))
        f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1],'wspace':0.05, 'hspace':0.05})

        for k in range(len(Y)):
            a0.text(Y[k, 0], Y[k, 1], labels_sub[k], bbox=dict(facecolor=cmap(plt_clr[labels_sub[k][0:3][0]])))
        a0.set_xlim((np.min(Y[:,0])-ep,np.max(Y[:,0])+ep))
        a0.set_ylim((np.min(Y[:,1])-ep,np.max(Y[:,1])+ep))
        a0.set_title('ICD codes distribution',fontsize=18)
        a0.set_xlabel('UMAP dim 1')
        a0.set_ylabel('UMAP dim 2')
        
        a1.scatter(Y[:, 0], Y[:, 1], c=freq_labels,s=50, cmap='seismic')
        a1.set_xlim((np.min(Y[:,0])-ep,np.max(Y[:,0])+ep))
        a1.set_ylim((np.min(Y[:,1])-ep,np.max(Y[:,1])+ep))
        a1.set_title('Rare (red) frequent (blue) codes',fontsize=18)
        a1.set_xlabel('UMAP dim 1')
        plt.show()
        plt.savefig('KAFE_UMAP.pdf', format='pdf', dpi=500,bbox_inches='tight')




fname = args.data + 'dummy.txt'
kafe = KAFEtrainer(fname,args)
kafe.train()
kafe.kafe_plot(100)
