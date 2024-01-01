
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0] // 2, 2, x.shape[1], x.shape[2]).transpose(1, 2)
        x = torch.relu(x.view(x.shape[0] * x.shape[2] * x.shape[3], x.shape[1])), 1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
