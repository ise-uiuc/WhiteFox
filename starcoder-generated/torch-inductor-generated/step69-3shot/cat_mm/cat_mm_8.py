
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=128, out_features=2, bias=False)
        self.drop = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        v = []
        for _ in range(10):
            v.append(self.lin(x1))
        return self.drop(torch.cat(v, 1))
# Inputs to the model
x1 = torch.randn(1024, 128)
