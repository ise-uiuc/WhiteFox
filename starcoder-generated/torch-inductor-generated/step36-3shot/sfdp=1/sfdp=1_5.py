
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        t1 = torch.matmul(query, key.transpose(-2, -1))
        t2 = t1.div(inv_scale_factor)
        t3 = t2.softmax(dim=-1)
        t4 = torch.nn.functional.dropout(t3, p=dropout_p)
        output = t4.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 128)
key = torch.randn(1, 8, 128)
value = torch.randn(1, 8, 128)
inv_scale_factor = 128 ** -0.5
dropout_p = 0.2
