
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x.reshape(-1, 3*x.shape[1])
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
