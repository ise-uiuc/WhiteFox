
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.relu(x) + torch.tanh(x)
        y = y.permute(1, 0, 2, 3).unsqueeze(1)
        y = torch.permute(y, (3,))
        y = y.permute(1, 0)
        y = y.permute(1, 0, 2).reshape(2, -1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4, 4)
