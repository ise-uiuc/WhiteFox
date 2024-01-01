
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=512, out_features=2048, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.nn.functional.hardswish(v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
