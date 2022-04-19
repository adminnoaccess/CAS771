from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal

# Loss functions
def loss_triteaching_cir(y_1, y_2, y_3, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    loss_3 = F.cross_entropy(y_3, t, reduction='none')
    ind_3_sorted = np.argsort(loss_3.cpu().data).cuda()
    loss_3_sorted = loss_3[ind_3_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # indices to update
    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    ind_3_update=ind_3_sorted[:num_remember].cpu()

    # if no update in indixes
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        ind_3_update = ind_3_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]])/float(num_remember)
    pure_ratio_3 = np.sum(noise_or_not[ind[ind_3_update]])/float(num_remember)

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_3_update], t[ind_3_update])
    loss_3_update = F.cross_entropy(y_3[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, torch.sum(loss_3_update)/num_remember, pure_ratio_1, pure_ratio_2, pure_ratio_3

def loss_triteaching_plus(logits, logits2, logits3, labels, forget_rate, ind, noise_or_not, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)
    outputs3 = F.softmax(logits3, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)
    _, pred3 = torch.max(logits3.data, 1)
    pred1 = pred1.cpu().numpy()
    pred2 = pred2.cpu().numpy()
    pred3 = pred3.cpu().numpy()
   
    # list of indices of disagreement
    disagree_id = []
    # mask for the list
    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
   
    # disagreement between each two
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx)

    for idx, p2 in enumerate(pred2): 
        if p2 != pred3[idx]:
            disagree_id.append(idx)

    for idx, p3 in enumerate(pred3): 
        if p3 != pred1[idx]:
            disagree_id.append(idx)

    # remove duplicates
    disagree_id = list(dict.fromkeys(disagree_id))   
    # make filter
    logical_disagree_id[disagree_id] = True
    
    # turn to logical_disagree_id from boolean to 0,1 form
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    # 1's in temp_disagree

    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
    
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id] 
        update_outputs3 = outputs3[disagree_id] 
        
        loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = loss_triteaching_cir(update_outputs, update_outputs2, update_outputs3, update_labels, forget_rate, ind_disagree, noise_or_not)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2
        update_outputs3 = outputs3

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)
        cross_entropy_3 = F.cross_entropy(update_outputs3, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
        loss_3 = torch.sum(update_step*cross_entropy_3)/labels.size()[0]
 
        pure_ratio_1 = np.sum(noise_or_not[ind])/ind.shape[0]
        pure_ratio_2 = np.sum(noise_or_not[ind])/ind.shape[0]
        pure_ratio_3 = np.sum(noise_or_not[ind])/ind.shape[0]
    return loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3  

