
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.ConvTranspose2d(7, 10, 1)
        self.t2 = torch.nn.Linear(10, 1)
    def forward(self, x1):
        v1 = self.t2(self.t1(x1))
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 15, 15)
