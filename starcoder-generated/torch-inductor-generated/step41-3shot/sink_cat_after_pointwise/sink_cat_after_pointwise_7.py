
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((torch.relu(x).relu(), torch.relu(x).relu()), dim=1)
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(2, 2, 2)
