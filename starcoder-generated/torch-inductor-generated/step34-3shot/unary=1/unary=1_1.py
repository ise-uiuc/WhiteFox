
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_1 = torch.nn.Linear(300, 500)
 
    def forward(self, x2):
        v233 = torch.nn.functional.linear(x2, self.Linear_1.weight, self.Linear_1.bias)
        v236 = v233 * 0.5
        v239 = torch.mul(v233, v233).mul(v233)
        v242 = v239 * 0.044715
        v248 = v239 * 0.7978845608028654
        v249 = torch.tanh(v248)
        v250 = v249 + 1
        v251 = v236 * v250
        return v251

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 300)
