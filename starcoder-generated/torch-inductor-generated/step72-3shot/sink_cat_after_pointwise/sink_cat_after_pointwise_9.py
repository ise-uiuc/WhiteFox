
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.randn(4) + torch.relu(x)
        x = x.unsqueeze(0)
        return x.view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
