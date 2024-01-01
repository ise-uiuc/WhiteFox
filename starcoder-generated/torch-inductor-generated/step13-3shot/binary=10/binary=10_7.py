
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=128, out_features=1024)
        self.linear2 = torch.nn.Linear(in_features=1024, out_features=1)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        v3 = self.linear2(v2)
        return v2, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 1)
__model__.eval()
