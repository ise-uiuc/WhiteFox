
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.sigmoid
    def forward(self, x3):
        v1 = torch.mm(x3, x3.T)
        v2 = torch.relu(v1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x3 = torch.randn(1, 8)
