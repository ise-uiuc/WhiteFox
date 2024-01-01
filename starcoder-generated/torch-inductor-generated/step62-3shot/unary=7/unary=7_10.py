
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 50, bias=False)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        max_l1 = torch.clamp(min=0, max=6, input=l1 + 3)
        l2 = l1 * max_l1
        l3 = l2 / 6
        return l3

# Initializing the model
m = CustomModel()

# Inputs to the model
x1 = torch.randn(3, 6, 5)
