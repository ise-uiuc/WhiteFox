
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1, bias=False)
        m = torch.tensor([[0.0, 1.0]] * 8)
        self.linear.weight.data.copy_(m)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 - 1.0
        y3 = y2.relu()
        return y3

# Initializing the model
m = Model()
torch.manual_seed(1341)

# Inputs to the model
x1 = torch.randn(2, 16)
