
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.dropout(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
