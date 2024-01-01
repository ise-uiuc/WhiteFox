
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.4)
    def forward(self, x):
        v1 = self.dropout(x)
        z1 = torch.rand_like(x) # random
        y1 = v1 + z1
        return y1

# Inputs to the model
x = torch.randn(1, 2)
