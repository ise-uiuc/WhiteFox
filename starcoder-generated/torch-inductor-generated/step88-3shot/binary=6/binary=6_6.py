
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        __init_variable_required_by_model_initialization__ = 24
        self.linear = torch.nn.Linear(__init_variable_required_by_model_initialization__, __init_variable_required_by_model_initialization__)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = 25 - v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
