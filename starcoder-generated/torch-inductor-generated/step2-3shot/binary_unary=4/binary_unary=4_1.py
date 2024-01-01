
class Model(nn.Module):
    def __init(self, tensor):
        super().__init__()
        self.linear = nn.Linear(16, 32, bias=True)
        self.tensor = tensor
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.tensor
        v3 = F.relu(v2)
        return v3

t = torch.zeros(32)
m = Model(t)

# Inputs to the model
x = torch.randn(1, 16)
