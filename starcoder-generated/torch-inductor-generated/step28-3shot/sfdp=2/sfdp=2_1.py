
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        
    def forward(self, query, key, value, scale_factor, dropout_p, mask):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = scaled_qk.div(scale_factor) 
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.drop_out(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value) 
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 8, 16) # Shape (B, S, T, D)
key = torch.randn(1, 2, 16, 32) # Shape (B, S, T, D)
value = torch.randn(1, 2, 16, 32) # Shape (B, S, T, D)
scale_factor = torch.randn(1, 128) # Shape (B, S)
dropout_p = 0.8
mask = torch.randn(1, 1, 16, 16) # Shape (B, S, S)
