
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 8
        self.linear_1 = torch.nn.Linear(n_feats, 16)
        self.linear_2 = torch.nn.Linear(16, 32)
        self.linear_3 = torch.nn.Linear(32, 16)
 
    def forward(self, x):
        h = self.linear_1(x)
        h = self.linear_2(h)
        h = self.linear_3(h)
        return h

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
