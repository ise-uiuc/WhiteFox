
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p, train=self.training)
        output = v4.matmul(value)
        return output

# Initializing the model
m = Model()

query = torch.randn(200, 512)
key = torch.randn(200, 512, 16)
value = torch.randn(200, 512, 16)
inv_scale_factor = torch.randn(200, 512, 16)
dropout_p = 0.1
