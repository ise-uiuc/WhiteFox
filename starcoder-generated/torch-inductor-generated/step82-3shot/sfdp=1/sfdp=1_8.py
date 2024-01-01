
class Model(torch.nn.Module):
    def __init__(self, hidden_dim, num_head, seq_length, dropout_p):
        super().__init__()
        self.scale_factor = (hidden_dim // num_head) ** -0.25 # Compute the value of the scale factor
        self.seq_length = seq_length # Store the sequence length
        self.dropout = torch.nn.Dropout(dropout_p) # Instantiate the dropout module
 
    def forward(self, q, k, v):
        softmax_qk = softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor, dim=-1) # Compute the scaled dot product and apply softmax to it
        output = self.dropout(softmax_qk).matmul(v) # Dropout and matmul applied in one line of code
        return output

# Initializing the model
self = Model(hidden_dim=32, num_head=2, seq_length=20, dropout_p=0.2)

# Inputs to the model
x1 = torch.randn(8, 20, 32)
x2 = torch.randn(8, 20, 32)
x3 = torch.randn(8, 20, 32)
