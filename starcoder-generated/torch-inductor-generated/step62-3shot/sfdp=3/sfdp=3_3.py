
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, __inputs__):
        v0 = torch.matmul(__inputs__.pop(), __inputs__.pop().transpose(-2, -1))
        v1 = v0 * 0.5
        v2 = torch.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.30000001192092896)
        v4 = torch.matmul(v3, __inputs__.pop())
        return v4

# Initializing the model
m1 = Model1()

# Inputs to the model
tensor = torch.randn(2, 2, 256, 256)
value = torch.randn(2, 28, 256)
