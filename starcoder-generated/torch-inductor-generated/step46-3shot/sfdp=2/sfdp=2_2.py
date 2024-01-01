
t1 = torch.softmax(torch.matmul(query, key.transpose(-2,-1))/inv_scale_factor, dim=-1)
t2 = torch.nn.functional.dropout(t1, p=dropout_p)
output = torch.matmul(t2, value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
x2 = torch.randn(1, 1, 1, 1)
