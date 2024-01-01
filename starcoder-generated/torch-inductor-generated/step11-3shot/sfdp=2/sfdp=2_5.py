
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / __scale_factor__
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, __dropout_p__)
        output = torch.matmul(v4, x3)
        return output
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25, 768)
x2 = torch.randn(1, 25, 768)
x3 = torch.randn(1, 25, 768)
