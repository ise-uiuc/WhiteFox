
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(256, 128)
 
    def forward(self, x1, **kwargs):
        x2 = self.linear(x1)
        return x2 + kwargs['x2_tensor'] if 'x2_tensor' in kwargs else x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 256)
x2 = torch.randn(2, 128)
