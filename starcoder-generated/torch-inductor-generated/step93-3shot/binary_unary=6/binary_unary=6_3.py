
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8, bias=False)
        self.other = torch.Tensor(np.random.randn(8, ).astype(np.float32))
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128)
m.eval()
