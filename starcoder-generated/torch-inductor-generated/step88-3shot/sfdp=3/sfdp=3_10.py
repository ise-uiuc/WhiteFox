
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(85, 64, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm1d(64)
        self.linear2 = torch.nn.Linear(64, 1, 1, stride=1, padding=0)
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=2)
        v2 = self.linear1(v1.view(1,6,64)).view(-1,1,64,1)
        v3 = self.bn(v2)
        v4 = self.linear2(v3.view(1,64)).view(-1,1,64,1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 1, 1)
k = torch.randn(6, 16, 1, 1)
v = torch.randn(6, 16, 1, 1)
dropout_p = 0.2
dropout_mask = m.generate_dropout_mask(q, k, v, dropout_p)
