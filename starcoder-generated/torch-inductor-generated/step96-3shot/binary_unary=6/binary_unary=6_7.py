
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
        self.const_1 = torch.tensor([0.32], requires_grad=requires_grad)
        self.const_2 = torch.tensor([0.578], requires_grad=requires_grad)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.const_1
        v3 = tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
