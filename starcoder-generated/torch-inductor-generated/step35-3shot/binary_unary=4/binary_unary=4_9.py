
class Model(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.linear.weight.data = torch.Tensor([weight], requires_grad=True)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3
 
# Initializing the model
m = Model(1.0)
print(m.state_dict())

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
