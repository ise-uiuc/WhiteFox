
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
    def forward(self,x1):
        v1 = self.relu2(x1)
        v2 = self.relu1(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 112, 112)
