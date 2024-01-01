
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=10175, out_features=2048, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        other = torch.randn(2048)
        other[:, 15] = 0
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v2)
        return v3
 
# Initializing the model
m = Model(num_classes=1000)
 
# Inputs to the model
x1 = torch.randn(1, 10175)
