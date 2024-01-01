
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
 
    def forward(self, x):
        v1 = self.layer(x)
        v2 = torch.nn.functional.dropout(v1, p=0.75)
        return v2.matmul(v1.transpose(-2, -1))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 1)
scale_factor = torch.randn(1, )
