
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=16, out_features=32, bias=True)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x2 = torch.randn([1, 16])
