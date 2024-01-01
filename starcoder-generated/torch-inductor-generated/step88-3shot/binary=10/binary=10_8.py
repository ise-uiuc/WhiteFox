
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2, bias=True)
 
    def forward(self, x1):
        x1 = x1.unsqueeze(2)
        x1 = torch.nn.functional.pad(x1, (3, 3, 0, 0))
        v1 = self.linear(x1)
        m = v1.size(0)
        v2 = torch.rand(m).unsqueeze(2).expand_as(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 158)
