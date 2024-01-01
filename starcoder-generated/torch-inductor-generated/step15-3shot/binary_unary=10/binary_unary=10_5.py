
class Model(torch.nn.Module):
    def __init__(self, dim, num_class):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(dim, num_class, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3
 
# Initializing the model
m = Model(1, 2)
x1 = torch.randn(3, 1)
x2 = torch.randn(3, 1)
