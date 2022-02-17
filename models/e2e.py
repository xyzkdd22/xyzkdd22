import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def NTXentLoss(a, b, tau=0.05):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)

    return torch.mean(neglog_num_by_den)


def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionMLP, self).__init__()
        hidden_dim = out_dim  # in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        return x


class E2E3DPretrained(nn.Module):

    def __init__(self, backbone, device, in_features=512, out_features=128, checkpoint=None):
        super(E2E3DPretrained, self).__init__()

        self.encoder = backbone
        self.encoder.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)

        # Set trainable parameters of the encoder off and train the MLP projection only
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.n_features = self.encoder.fc.in_features
        self.out_features = out_features

        self.encoder.fc = ProjectionMLP(self.n_features, self.out_features)

    def forward(self, x):
        z = self.encoder(x)
        return z


class E2E3DScratch(nn.Module):

    def __init__(self, backbone, num_classes, out_features=128, family='resnet'):
        super(E2E3DScratch, self).__init__()

        if family == "efficient":
            self.encoder = backbone
            self.n_features = self.encoder.classifier[1].in_features
            self.out_features = out_features

            self.encoder.classifier = ProjectionMLP(self.n_features, self.out_features)
        elif family == "resnet":
            self.encoder = backbone
            # self.encoder.fc.out_features = num_classes
            self.n_features = self.encoder.fc.in_features

            self.out_features = out_features

            self.encoder.fc = ProjectionMLP(self.n_features, self.out_features)
            # dim_mlp = self.encoder.fc.weight.shape[1]
            # self.projector = nn.Sequential(nn.Linear(dim_mlp, out_features),
            #                                 nn.ReLU(),
            #                                 nn.Linear(out_features, out_features))
        else:
            raise KeyError("No family is used as {}, supports only 'efficient' or 'resnet'".format(family))

    def forward(self, x):
        z = self.encoder(x)
        # h = self.projector(z)
        return z

if __name__ == '__main__':
    from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18

    encoder = mc3_18(pretrained=False)
    m = E2E3DScratch(encoder, num_classes=101).cuda()
    print(m)
    print('Total trainable parameters are: {}'.format(sum(p.numel() for p in m.parameters() if p.requires_grad)))
    x = torch.randn([4, 3, 4, 32, 32]).cuda()
    y = torch.rand_like(x)
    o_x = m.encoder(x)
    o_y = m(y)
    print("Output of x: {}, Output of y: {}".format(o_x.size(), o_y.size()))