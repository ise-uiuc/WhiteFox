
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(256, 256, True)
        self.act = torch.nn.Sigmoid()
 
    def forward(self, input_tensor, other_tensor):
        v1 = self.f1(input_tensor)
        v2 = v1 + other_tensor
        v3 = self.act(v2)
        return v3, v2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(32, 256)
other = torch.randn(32, 256)
__output__, __output2__ = m(input, other)


