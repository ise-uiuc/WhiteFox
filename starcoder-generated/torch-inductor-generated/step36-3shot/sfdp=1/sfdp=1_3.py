
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dot = torch.nn.Linear(127, 64)
 
    def forward(self, x1, x2):
        query = x1
        key = x2
        v1 = self.dot(query)
        v2 = self.dot(key)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(v3.shape[-1])
        v4 = v3.div(inv_scale_factor)
        v5 = v4.softmax(dim=-1)
        dropout_p = 0.01
        v6 = torch.nn.functional.dropout(v5, p=dropout_p)
        value = x2
        v7 = torch.matmul(v6, value)
        return v7

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 128, 127)
x2 = torch.randn(1, 128, 64)
