
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x = x.squeeze()
        return x.squeeze(1).tanh()
# Inputs to the model
x = torch.randn(1, 3, 1)
