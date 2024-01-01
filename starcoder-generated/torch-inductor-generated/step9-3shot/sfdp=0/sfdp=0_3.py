
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_linear = torch.nn.Linear()
 
    def forward(self, x1, x2):
        v1 = self.input_linear(x1)
        v2 = torch.transpose(v1, -2, -1)
        v3 = torch.matmul(v1, v2)
        v4 = 1 / np.sqrt(x1.shape[-1])
        v5 = torch.matmul(v3, v4)
        v6 = torch.transpose(v5, -2, -1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 512)
x2 = torch.randn(1, 16, 512)
