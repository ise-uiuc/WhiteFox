
class Model(torch.nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        
        self.to_qkv = torch.nn.Linear(hidden, hidden*3, bias=False)
        self.q_dropout = torch.nn.Dropout(p=0.1)
        self.k_dropout = torch.nn.Dropout(p=0.1)
        self.v_dropout = torch.nn.Dropout(p=0.1)
        self.scale_factor = 1 / (hidden)**0.5
        
        self.output = torch.nn.Linear(hidden, hidden)
     
        self.softmax = torch.nn.Softmax(dim=-1)
     
    def forward(self, x):
        b, t, e = x.size()
        assert e == self.hidden
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, h, t.size(1) // h, e), qkv)
        
        q = self.q_dropout(q)
        k = self.k_dropout(k)
        v = self.v_dropout(v)
        
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.mul(self.scale_factor) # Scale the dot product by a factor
        softmax_qk = self.softmax(scaled_qk) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        
        output = output.reshape(b, t, h, -1).transpose(1, 2)[..., 0] # Merge the heads out and transpose back
        return self.output(output)

# Initializing the model
m = Model(hidden=32, heads=4)

# Inputs to the model
x = torch.randn(1, 4, 32)
