
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 10)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs["other"]
        return v2

# Initializing the model
m = Model()

# Inputs for the model
x1 = torch.randn(1, 2)
other = torch.randn(1, 10)
kwargs = {"other": other}
