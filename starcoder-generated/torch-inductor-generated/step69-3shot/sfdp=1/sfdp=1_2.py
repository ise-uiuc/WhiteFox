
class Model(torch.nn.Module):
    def __init__(self, dropout_p, inv_scale_factor):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        s1 = torch.matmul(query, key.transpose(-2, -1))
        s2 = s1.div(self.inv_scale_factor)
        s3 = s2.softmax(dim=-1)
        s4 = torch.nn.functional.dropout(s3, p=self.dropout_p)
        v1 = torch.matmul(s4, value)
        return v1

# Initializing the model
m = Model(0.0, 1.0)

# Inputs to the model
query = torch.randn(1, 12, 512)
key = torch.randn(1, 24, 1024)
value = torch.randn(1, 24, 1024)
