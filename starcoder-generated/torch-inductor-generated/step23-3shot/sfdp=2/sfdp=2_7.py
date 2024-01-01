
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = v4.matmul(x1)
        v6 = v5 + x2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
input_query = torch.randn(96, 8, 512)
input_key = torch.randn(96, 8, 512)
input_value = torch.randn(96, 8, 512)
x1, x2, x3 = input_query, input_key, input_value 
