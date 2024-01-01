
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        return torch.matmul(torch.transpose(x, 1, 0), x)
# Inputs to the model
x = torch.randn(1, 2)
