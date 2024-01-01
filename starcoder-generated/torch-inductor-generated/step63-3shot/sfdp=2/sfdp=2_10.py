
class Model(torch.nn.Module):
def __init__(self, d_query, d_key, d_value, dropout_p):
super(Model, self).__init__()
self.dropout_p = dropout_p
self.softmax = nn.Softmax(dim=-1)
self.query_linear = nn.Linear(d_query, d_key)
self.key_linear = nn.Linear(d_key, d_key)
self.value_linear = nn.Linear(d_value, d_value)
self.dropout = nn.Dropout(dropout_p)
self.scale_factor = sqrt(d_key)
 
def forward(self, q, k, v):
q = self.query_linear(q)
k = self.key_linear(k)
v = self.value_linear(v)
dot_products = torch.matmul(q, k.transpose(1, 2))
scaled_products = dot_products / self.scale_factor
softmax_products = self.softmax(scaled_products)
dropout_products = self.dropout(softmax_products)
output = torch.matmul(dropout_products, v)
return output

# Initializing the model:
# You can generate the data for the model with the following lines of code.
d_query = 3
d_key = 7
d_value = 6
dropout_p = 0.25
q = torch.rand(2, d_query)
k = torch.rand(3, d_key)
v = torch.rand(5, d_value)
