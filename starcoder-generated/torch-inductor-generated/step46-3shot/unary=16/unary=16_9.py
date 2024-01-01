
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(8, 15, bias=True)
 
    def forward(self, x1):
        x0 = x1
        x1 = self.conv(x1)
        x1 = F.relu(x1)
        return


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
