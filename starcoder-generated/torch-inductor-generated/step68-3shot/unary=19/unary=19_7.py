
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc7 = torch.nn.Linear(in_features = 512, out_features = 4096, bias = True)
        self.fc8 = torch.nn.Linear(in_features = 4096, out_features = 4096, bias = True)
 
    def forward(self, x1):
        v1 = self.fc7(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 1, 1)
