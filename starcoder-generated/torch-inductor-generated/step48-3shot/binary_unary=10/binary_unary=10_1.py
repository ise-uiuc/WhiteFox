
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(8 * 8 * 1, 128), torch.nn.Linear(128, 10)])
 
    def forward(self, x1):
        v1 = x1.reshape(x1.shape[0], -1)
        v2 = self.linears[0](v1)
        v3 = self.linears[1](v2)
        return v3
 
m = Model().eval()

# The first input tensor for the model
x1 = torch.randn(1, 8, 8, 1)
# The second input tensor for the model
x2 = torch.randn(1, 128)
# The target output tensor for the model
y = torch.randn(1, 10)

# Input vectors
X = [x1, x2]

# Loss function
loss_func = torch.nn.MSELoss()

# The initial output of the model given input vectors `X`
Y = m(*X)

