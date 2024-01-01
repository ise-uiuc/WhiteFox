
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, __input__0, __input__1):
        v0 = torch.matmul(__input__0, __input__1.transpose(-2, -1))
        v1 = v0 * 0.125
        v2 = F.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.1, training=True)
        output = torch.matmul(v3, __input__0)
        return output

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(32, 32, 64)
x2 = torch.randn(32, 64, 64)
