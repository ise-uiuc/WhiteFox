
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.mean(0, keepdim=True)
        z = torch.relu(y)
        y = y.view(-1)
        return y
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
