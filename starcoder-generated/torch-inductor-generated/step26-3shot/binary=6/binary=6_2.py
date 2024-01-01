
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 9, bias=True) # A linear transformation is applied to an input tensor, where the input tensor's last dimension is 1 and the output tensor's last dimension is 9
        self.constant = torch.nn.Parameter(torch.ones(1, 1, 1, 9)) # A learned parameter is used to add to the output of the linear transformation
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.constant # v2 is the output of the linear transformation, and self.constant is the known parameter that is subtracted from the output of the linear transformation; in this case, we want to get v2 equals to self.constant because the subtraction only involves the known parameter of the model
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 5, 9)
