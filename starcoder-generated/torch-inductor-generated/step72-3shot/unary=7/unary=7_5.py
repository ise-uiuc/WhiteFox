
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        h1 = self.linear(x)
        h2 = h1 * torch.clamp(h1 + 3, min=-1, max=14)
        h3 = h2 / 12
        return h3

# Input to the model
x = torch.randn(1, 8)
