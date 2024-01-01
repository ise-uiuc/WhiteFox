
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1)
        return torch.nn.functional.relu(torch.nn.functional.tanh(x))
# Inputs to the model
x = torch.randn(2, 3, 4)
