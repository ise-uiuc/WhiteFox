
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 200)
 
    def forward(self, x1, other_param):
        v1 = self.linear(x1)
        v2 = v1 + other_param
        return v2

# Initializing the model
m = Model()

# Inputs to the model, "other_param" should be a PyTorch tensor of any shape and dtype, with the same shape and dtype as the output of the linear transformation
x1 = torch.randn(1, 10)
__other_param__ = torch.randn(1, 200)
