
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(16, 256, bias=True)
        self.linear_2 = torch.nn.Linear(256, 64, bias=True)
        self.linear_3 = torch.nn.Linear(64, 8, bias=True)
 
    def forward(self, x1):
        out = self.linear_1(x1)
        out = torch.sigmoid(out)
        out = self.linear_2(out)
        out = torch.sigmoid(out)
        out = self.linear_3(out)
        out = torch.sigmoid(out)
        return out
 
m = Model()

# Inputs to the model
x1 = torch.randn(16)
