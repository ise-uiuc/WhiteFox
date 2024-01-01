
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.layers(x)
        x = self.dropout(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.unsqueeze(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 2)
