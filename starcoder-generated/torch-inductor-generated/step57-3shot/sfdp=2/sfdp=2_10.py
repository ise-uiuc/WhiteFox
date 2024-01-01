
class Model(torch.nn.Module):
    def __init__(self,
                 query_size = 512,
                 key_size = 512,
                 value_size = 512,
                 inv_scale_factor = 1 / 2**0.5,
                 dropout_p = 0.75):
        super().__init__()
        self.query = torch.nn.Linear(query_size, key_size)
        self.value = torch.nn.Linear(value_size, key_size)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2, x3):
        v1 = self.query(x1)
        v2 = self.value(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(self.inv_scale_factor)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=self.dropout_p)
        v7 = torch.matmul(v6, x3)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 512)
x3 = torch.randn(1, 512)
