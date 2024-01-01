
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor, *args, other):
        x = input_tensor + other
        return x
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.custom = CustomModule()
 
    def forward(self, x1):
        v1 = self.custom(x1, other=0.75)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
