
class Model(nn.Module):
    def __init__(self, other):
        super(Model, self).__init__()
        self.other = other

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3
    
# Initializing the model
m = Model(torch.randn(256))

# Input to the model
x1 = torch.randn(5, 256)
