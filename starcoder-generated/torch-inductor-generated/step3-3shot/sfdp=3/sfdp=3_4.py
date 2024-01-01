
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.125
        v3 = torch.nn.functional.softmax(v2, -1)
        v4 = torch.nn.functional.dropout(v3)
        output = v4.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 768, 8)
x2 = torch.randn(5, 8, 768)
