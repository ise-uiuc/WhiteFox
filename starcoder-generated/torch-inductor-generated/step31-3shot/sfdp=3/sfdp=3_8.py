
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input1, input2, input3):
        v1 = torch.matmul(input1, input2.transpose(-2, -1))
        v2 = v1.mul(10)
        v3 = x3.softmax(-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        return v4.matmul(input3)

# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(50, 64)
input2 = torch.randn(64, 80)
input3 = torch.randn(80, 10)
