
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(1024, 1024)
        self.k = torch.nn.Linear(1024, 1024)
        self.v = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.3)
 
    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        qk = torch.matmul(q, k.transpose(-1, -2)) # Compute the dot product of the query and key tensors
        inv_scale_factor = 1 / math.sqrt(self.k.in_features) # Specify the inverse scale factor using an approximation
        softmax_qk = qk.div(inv_scale_factor).softmax(dim=-1) # Scale the dot product by the inverse scale factor, then apply softmax to the scaled dot product tensor
        dropout_qk = self.dropout(softmax_qk) # Apply dropout to the softmax output  
        output = torch.matmul(dropout_qk, v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 1024)
x3 = torch.randn(1, 1024)
