
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = torch.matmul(x1.permute(...,...,...), x2.permute(...,...,...))
        return v0
# Inputs to the model
x1 = torch.randn(in_features_0,...,...,...) # in_features_0 > in_features_1
x2 = torch.randn(in_features_1,...,...,...) # in_features_1 > in_features_0
