
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1.view(x1.size(0), 1, -1), x1.view(x1.size(0), -1, 1)).view(x1.size(0), -1)
        v2 = v1 + x1.view(-1).unsqueeze(-1)
        v3 = torch.nn.functional.relu(v2.view(-1)).view(x1.size())
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
