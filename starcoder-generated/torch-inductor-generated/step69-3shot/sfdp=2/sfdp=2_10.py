
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qk, inv_scale_factor, dropout_p):
        t1 = qk.div(inv_scale_factor)
        t2 = t1.softmax(dim=-1)
        t3 = torch.nn.functional.dropout(t2, p=dropout_p)
        t4 = t3.matmul(t2)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 20, 768)
key = torch.randn(5, 10, 768)
value = torch.randn(5, 10, 768)
inv_scale_factor = 2.0
dropout_p = 0.2
