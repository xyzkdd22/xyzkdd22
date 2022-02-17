from operator import mod
from time import sleep
import torch
import torch.nn as nn
from torch.nn import functional as F

def ctr(q, k, criterion, tau, device):
    logits = torch.mm(q, k.t())
    N = q.size(0)
    labels = range(N)
    labels = torch.LongTensor(labels).to(device)
    loss = criterion(logits/tau, labels)
    return 2*tau*loss


class MoCoV3(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, K=4096, m=0.999, T=0.07, input_features= 512, output_features=128, num_classes=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoV3, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.num_classes = num_classes

        # create the encoders
        self.encoder_q = nn.Sequential(*list(base_encoder.children())[:-1])
        self.encoder_k = nn.Sequential(*list(base_encoder.children())[:-1])
        
        self.classifier_q = nn.Linear(in_features=input_features, out_features=self.num_classes)
        self.classifier_k = nn.Linear(in_features=input_features, out_features=self.num_classes)

        self.projection_q = nn.Sequential(nn.Linear(input_features, output_features), nn.ReLU(), nn.Linear(output_features, output_features))
        self.projection_k = nn.Sequential(nn.Linear(input_features, output_features), nn.ReLU(), nn.Linear(output_features, output_features))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_pro_q, param_pro_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_pro_k.data.copy_(param_pro_q.data)  
            param_pro_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_pro_q, param_pro_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_pro_k.data = param_pro_k.data * self.m + param_pro_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1,q2,k1,k2
        """

        # compute query features
        q1, q2 = self.projection_q(self.encoder_q(x1).view(x1.size(0), -1)), self.projection_q(self.encoder_q(x2).view(x2.size(0), -1))  # [B, 128]
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1, k2 = self.projection_k(self.encoder_k(x1).view(x1.size(0), -1)), self.projection_k(self.encoder_k(x2).view(x2.size(0), -1))  # keys: NxC  => [B, 400]
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
        
        h_i = self.classifier_q(self.encoder_q(x1).view(x1.size(0), -1))
        h_j = self.classifier_k(self.encoder_q(x2).view(x2.size(0), -1))

        return q1, q2, k1, k2, h_i, h_j

# if __name__ == '__main__':
#     from torchvision.models import video

#     base_encoder = video.r2plus1d_18(pretrained=True)
#     x1 = torch.randn((16, 3, 16, 112, 112)).cuda()
#     x2 = torch.randn_like(x1).cuda()

#     model = MoCoV3(base_encoder=base_encoder, num_classes=10).cuda()
#     # print(model)
#     q1, q2, k1, k2, y_q, y_k = model(x1, x2)
#     print("q1: {}, q2: {}, k1: {}, k2: {}, y_q: {}, y_k: {}".format(
#         q1.size(), q2.size(), k1.size(), k2.size(), y_q.size(), y_k.size()
#     ))
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     criterion = nn.CrossEntropyLoss().cuda()
#     K, m, T = 4096, 0.999, 0.07
#     loss = ctr(q1, k2, criterion, T, device) + ctr(q2, k1, criterion, T, device)
#     print("Loss: {}".format(loss))
#     print('Total trainable parameters are: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))