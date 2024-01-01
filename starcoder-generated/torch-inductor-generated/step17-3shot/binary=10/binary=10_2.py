
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2
# Inputs to the model with a dummy tensor
x = torch.randn(1, 128)
other = torch.randn(1, 32)
