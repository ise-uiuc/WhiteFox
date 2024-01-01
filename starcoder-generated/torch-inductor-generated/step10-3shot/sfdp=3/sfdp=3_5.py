
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.02)
 
    def forward(self, __input__):
        out1 = torch.matmul(__input__, __input__.transpose(-2, -1))
        out2 = out1.mul(0.02)
        out3 = torch.nn.functional.softmax(out2, dim=-1)
        out4 = self.dropout(out3)
        out5 = torch.matmul(out4, __input__)
        return out5

# Initializing the model
m = Model()

# Inputs to the model
__inputs__ = torch.randn(1, 32, 64)
