
class Model(torch.nn.Module):
    def __init__(self, num_in, num_features, num_out, num_hidden):
        super().__init__()
        self.linear1 = torch.nn.Linear(num_in, num_features)
        self.linear2 = torch.nn.Linear(num_features, num_out)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model(num_in=1, num_features=8, num_out=4, num_hidden=2)

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 8)
