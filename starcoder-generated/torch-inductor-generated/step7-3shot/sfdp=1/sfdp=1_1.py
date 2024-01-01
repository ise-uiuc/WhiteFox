
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(32, 32)
        self.wk = torch.nn.Linear(32, 32)
        self.wv = torch.nn.Linear(32, 32)
        self.scale_factor = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.dropout_p = 0.5
 
    def forward(self, q, k, v):
        q = self.wq(q) # Apply query linear transformation
        k = self.wk(k) # Apply key linear transformation
        v = self.wv(v) # Apply value linear transformation
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.scale_factor) # Divide the dot product by the scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(8, 64, 32)
k = torch.randn(8, 64, 32)
v = torch.randn(8, 64, 32)
