
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU(), torch.nn.Linear(3, 3), torch.nn.Sigmoid())
        return m(x)
# Inputs to the model
x = torch.randn(2)
