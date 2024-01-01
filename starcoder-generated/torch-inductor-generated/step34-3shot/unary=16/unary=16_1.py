
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu(v1)
        return v2

# Input to the model
x1 = torch.randn(8, 128)
