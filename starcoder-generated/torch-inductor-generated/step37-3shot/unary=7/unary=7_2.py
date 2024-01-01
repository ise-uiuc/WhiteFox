
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(min=0, max=6, v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
input_size = {}
input_size['input'] = [1, 128]
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
