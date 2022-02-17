import torch
import torch.nn as nn
import torch.nn.functional as F

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128, num_layers=3):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = num_layers

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SemiPretrained(nn.Module):

    def __init__(self, backbone, device, num_classes=10, checkpoint=None):
        super(SemiPretrained, self).__init__()

        self.encoder = backbone
        self.encoder.fc.out_features = num_classes
        self.encoder.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)

        # Set trainable parameters of the encoder off and train the MLP projection only
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        return z


class SemiScratch(nn.Module):

    def __init__(self, backbone, num_classes=10, family='resnet'):
        super(SemiScratch, self).__init__()

        if family == "efficient":
            self.encoder = backbone
            self.encoder.fc.out_features = num_classes

        elif family == "resnet":
            self.encoder = backbone
            dim_mlp = self.encoder.fc.weight.shape[1] # 512
            # global average pooling and classifier
            # self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.001)
            # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            # self.avgpool = nn.AvgPool2d(8)
            # self.fc = nn.Linear(n_channels[3], num_classes)
            self.encoder.fc = nn.Linear(dim_mlp, num_classes)

        else:
            raise KeyError("No family is used as {}, supports only 'efficient' or 'resnet'".format(family))

    def forward(self, x):
        z = self.encoder(x)
        return z

class ContrastiveFixMatch(nn.Module):

    def __init__(self, backbone, input_features= 512, output_features=128, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        self.classifier = nn.Linear(in_features=input_features, out_features=num_classes)

        self.projection = projection_MLP(in_dim=input_features, out_dim=output_features)
        self.predictor = prediction_MLP(in_dim=output_features, out_dim=output_features)

    def forward(self, x):
        h = self.encoder(x)
        y = self.classifier(h.view(x.size(0), -1))
        z = self.projection(h.view(x.size(0), -1))
        p = self.predictor(z)
        return z, p, y


# if __name__ == '__main__':
#     from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
#     encoder = r2plus1d_18(pretrained=False)
#     cm = ContrastiveFixMatch(encoder).cuda()
#     print("\n\n", cm)
#     print('Total trainable parameters are: {}'.format(sum(p.numel() for p in cm.parameters() if p.requires_grad)))
#     x = torch.randn([4, 3, 4, 32, 32]).cuda()
#     y = torch.rand_like(x)
#     z1, p1, y1 = cm(x)
#     z2, p2, y2 = cm(y)
    
#     L = D(p1, z2) / 2 + D(p2, z1) / 2
#     print("\n\nLoss L: {}, Output of y_1: {}, output of y_2: {}".format(L, y1.size(), y2.size()))