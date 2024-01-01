
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=1905, out_features=256, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)
        self.drop = torch.nn.Dropout(p=0.2496)
    def forward(self, x1):
        v = self.lin(x1)
        v = self.relu(v)
        v = self.drop(v)
        v = self.lin(v)
        v = self.relu(v)
        v = self.drop(v)
        return v
# Inputs to the model
x1 = torch.randn(1, 1905, device='cuda')
