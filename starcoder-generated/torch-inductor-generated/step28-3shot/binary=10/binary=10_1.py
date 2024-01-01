
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(17, 10)
        self.linear.bias = torch.nn.Parameter(torch.tensor([0.909703, -0.011381, -0.037906, 1.987684, -1.719269, -0.758099, 0.418843, -0.735989, 1.524456, -1.688067]))
        self.linear.weight = torch.nn.Parameter(torch.tensor([-0.240726, -0.044200, -0.540115, -0.269951, -0.476428, 0.091757, -0.781907, -0.409872, 0.303320, -0.425438, 0.890566, 0.881638, 0.463365, -0.638760, 0.092376, 0.230328, -0.570235, -0.504842, -1.309621, -0.022977, 0.087456, 0.637414]))
        self.linear.weight.requires_grad = False
 
    def forward(self, input_tensor, other):
        return self.linear(input_tensor) + other

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(7, 17)
other = torch.randn(7, 10)
