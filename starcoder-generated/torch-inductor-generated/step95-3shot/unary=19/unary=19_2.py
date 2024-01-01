
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_in):
        v1 = torch.matmul(x_in, torch.tensor(np.random.uniform(-1.0, 1.0, [128, 128])
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x_in = torch.randn(128)
