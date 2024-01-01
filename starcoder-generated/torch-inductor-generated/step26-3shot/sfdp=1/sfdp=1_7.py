
class Model(torch.nn.Module):
   def forward(self, query, key, value, scale_factor, dropout_p):
       q1 = torch.matmul(query, key.transpose(-2, -1))
       v1 = q1.div(scale_factor)
       v2 = v1.softmax(dim=-1)
       v3 = torch.nn.functional.dropout(v2, p=dropout_p)
       output = v3.matmul(value)
       return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 1024, 256)
key = torch.randn(3, 1024, 256)
value = torch.randn(3, 2048, 256)
scale_factor = torch.randn(3, 256, 256)
dropout_p = 0.4
