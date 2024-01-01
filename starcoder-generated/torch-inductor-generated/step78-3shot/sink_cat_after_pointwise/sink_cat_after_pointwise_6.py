
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.relu(x).relu()
        return x
# Inputs to the model
x = torch.randn(3, 4)
