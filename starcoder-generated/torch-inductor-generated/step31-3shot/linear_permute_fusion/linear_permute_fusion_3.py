
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = torch.nn.Linear(2, 2, 1e-3)
    def forward(self, x1, x2):
        v1a = torch.nn.functional.linear(x1, x2)
        v1b = torch.nn.functional.linear(x2, x1[:1])
        v1c = v1a.expand(v1b.size()) + v1b
        return v1c
# Inputs to the model
