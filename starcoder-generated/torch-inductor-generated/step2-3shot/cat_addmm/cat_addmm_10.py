
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m0 = torch.nn.Linear(10, 20)
        self.m1 = torch.nn.Linear(20, 10)
        self.b1 = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v0 = torch.relu(self.m0(x1))
        v1 = self.m1(v0)
        v1 = v1 + self.b1(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
__input_size__ = __torch__.Size([[512]])
__output_size__ = __torch__.Size([[__batch_size__, 512]])
x1 = torch.randn(__input_size__)
