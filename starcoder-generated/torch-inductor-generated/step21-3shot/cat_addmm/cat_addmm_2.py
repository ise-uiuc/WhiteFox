
# inputs: in_features->16, out_features->64, kernel_size->3
# inputs: in_features->10, out_features->20
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(64, out_features=1, bias=False)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 64)
