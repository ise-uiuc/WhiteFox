
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 12)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.layers(x)
        x = self.dropout(x)
        return x
# Inputs to the model
x = torch.randn(8, 8)
