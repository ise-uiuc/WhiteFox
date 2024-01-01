
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x2 = x1.unsqueeze(0)
        v1 = self.sigmoid(self.linear(x2)).squeeze()
        return v1
# Inputs to the model
x1 = torch.randn(2, 2, device='cpu')
