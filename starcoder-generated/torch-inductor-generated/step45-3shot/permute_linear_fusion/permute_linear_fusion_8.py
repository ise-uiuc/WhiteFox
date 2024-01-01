
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = torch.nn.Unsqueeze(1)
        self.linear = torch.nn.Linear(2, 2)
        self.permute = torch.nn.Unsqueeze(2)
    def forward(self, x1):
        v1 = self.squeeze(x1)
        v2 = self.linear(v1)
        return self.permute(v2)
# Inputs to the model
x1 = torch.randn(1, 2)
