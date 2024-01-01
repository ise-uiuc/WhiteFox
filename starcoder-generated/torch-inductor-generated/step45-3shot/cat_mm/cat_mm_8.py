
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def generate_t2(self, x):
        return torch.cat([x, x, x], 0)
    def forward(self, x):
        t1 = self.generate_t2(x)
        t2 = torch.cat([t1, t1, t1], 0)
        return torch.cat([t2, t2], 0)
# Input to the model
x = torch.randn(8, 4)
# Model end
