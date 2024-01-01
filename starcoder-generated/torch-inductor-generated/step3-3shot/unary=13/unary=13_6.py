
class Model(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linear = torch.nn.Linear(in_feature, out_feature)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
in_feature, out_feature = 100, 100
m = Model(in_feature, out_feature)

# Inputs to the model
x1 = torch.randn(1, in_feature)
