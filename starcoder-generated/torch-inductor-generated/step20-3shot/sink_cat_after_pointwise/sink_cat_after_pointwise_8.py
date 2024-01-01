
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.relu(x)
        y2 = torch.sigmoid(x)
        return x + y1
# Inputs to the model
x = torch.randn(2, 3)
