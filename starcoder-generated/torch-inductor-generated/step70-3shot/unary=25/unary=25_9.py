
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 19)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = torch.empty_like(v1, dtype = v1.dtype).fill_(-0.01)
        v4 = torch.where(v2, v1, v3) # The negative slope in this model is set to be -0.01
        return v4

# Initializing the model
m = Model()

# Inputs to the model
inputshape = [1,10]
x1 = torch.randn(inputshape)
