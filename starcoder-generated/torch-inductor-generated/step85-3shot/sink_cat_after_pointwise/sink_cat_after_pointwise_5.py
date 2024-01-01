
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = x.view(x.size(0), -1)
        t1 = t0.flatten()
        t1 = torch.unsqueeze(t0, 1).float()
        return torch.relu(t0)
# Inputs to the model
x = torch.randn(1, 1, 1)
