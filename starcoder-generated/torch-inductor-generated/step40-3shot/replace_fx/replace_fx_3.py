
class m1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    def forward(self, inputs):
        x1 = self.layers(inputs)
        x2 = torch.nn.functional.softmax(x1, dim=0)
        x3 = torch.rand_like(x1, dim=0)
        return x2 + x3
class m2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        x1 = torch.nn.functional.dropout(inputs, p=0.5)
        x2 = x1.sigmoid()
        x3 = torch.rand_like(x1.sigmoid())
        return x2 - x3
m3 = m1()
m4 = m2()
def forward(inputs):
    x1 = m3(inputs)
    x2 = x1.softmax()
    return x2 - m4(x2)
# Inputs to the model
inputs = torch.randn(128, 64)
# Outputs of the model
torch.manual_seed(0) # Set random seed to 0 for recording the model graph
o = forward(inputs)
