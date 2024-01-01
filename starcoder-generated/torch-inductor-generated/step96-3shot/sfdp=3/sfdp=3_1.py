
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input1, input2, input3, beta1):
        v7 = torch.matmul(input1, input3.transpose(-2, -1))
        v8 = v7 * beta1
        v9 = v8.softmax(dim=-1)
        v10 = torch.nn.functional.dropout(v9, p=0.1)
        v11 = v10.matmul(input2)
        return v11

# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(10, 30, 256) # query tensor
input2 = torch.randn(10, 512, 256) # value tensor
input3 = torch.randn(30, 512) # key tensor
beta = torch.randn(1, 1) # scale factor
