
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def __scaled_dot_product(self, v1, v2):
        v12 = v1 @ v2.transpose(-2, -1)
        return v12.float() / math.sqrt(v1.size(-1))
 
    def forward(self, x1, x2, x3):
        a1 = self.__scaled_dot_product(x1, x2)
        a2 = a1 + x3
        a3 = torch.softmax(a2, dim=-1)
        output = a3 @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 6, 512)
x2 = torch.randn(8, 10, 512)
x3 = torch.randn(8, 6, 10)
