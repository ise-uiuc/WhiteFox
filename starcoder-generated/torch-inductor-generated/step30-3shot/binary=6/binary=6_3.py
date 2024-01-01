
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model(16, 16)

# Inputs to the model
x1 = torch.randn(1, 16)
__other = torch.FloatTensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]]])
