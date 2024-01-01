
class Model(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = torch.nn.Linear(in_ch, out_ch)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(3, 2)

# Inputs to the model
x1 = torch.randn(1, 3)
