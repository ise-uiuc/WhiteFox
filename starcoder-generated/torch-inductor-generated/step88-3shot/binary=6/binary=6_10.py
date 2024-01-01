
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 16)
 
    def forward(self, x1, x2, x3):
        v3 = torch.cat([x1, x2, x3], dim=1)
        v5 = self.linear1(v3)
        v6 = self.linear2(v5)
        v7 = v6 - 10 # v6 equals -10
        v10 = self.linear2(v7)
        return v10
 
# Initializing the model
m = ExampleModel()

# Inputs to the model

x1 = torch.randn(1, 32, 7, 7)
x2 = torch.randn(2, 8, 32, 32)
x3 = torch.randn(3, 9, 13)

