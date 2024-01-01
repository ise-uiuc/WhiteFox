
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
        self.bn = torch.nn.BatchNorm2d(16)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = self.bn(v2)
        return v3

# Initializing the model
def init_model():
    m = Model()
    print(m.state_dict().keys())
    print("Model is: ", m)
    return m

init_model()

# Inputs to the model
x = torch.randn(1, 3)
other = torch.randn(1, 16)
