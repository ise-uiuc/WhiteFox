
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 8)
        self.linear_2 = torch.nn.Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = torch.clamp_min(v1, 0.01)
        v3 = torch.clamp_max(v2, 0.10710678118654757)
        outputs = self.linear_2(v3)
        return outputs

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
