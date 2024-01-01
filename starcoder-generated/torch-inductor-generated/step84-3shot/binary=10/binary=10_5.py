
class Model(torch.nn.Module):
    def __init__(self):
        super(MyModuleClass, self).__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randint(10, size=(5,5,5)).float()
other = torch.rand(1).item()
