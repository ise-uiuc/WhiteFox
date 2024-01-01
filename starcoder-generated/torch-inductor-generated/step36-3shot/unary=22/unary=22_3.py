
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 256)
        self.linear2 = torch.nn.Linear(256, 512)
 
    def forward(self, x1):
        x = self.linear1(x1)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
__input_tensor__ = torch.randn(1, 64)
