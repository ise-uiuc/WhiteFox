
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(2, 1000)
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.8) + torch.nn.functional.dropout(x1, p=0.7)
        x2 = torch.tanh(x2)
        x3 = torch.nn.functional.dropout(x2, p=0.5)
        x4 = self.l0(x3)
        x5 = x4.mean()
        return torch.cos(x5) + torch.sigmoid(x5) - torch.sin(x5)
# Inputs to the model
x1 = torch.randn(2, 2)
