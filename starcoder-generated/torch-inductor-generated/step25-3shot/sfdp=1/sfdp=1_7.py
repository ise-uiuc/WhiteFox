
class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):         
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)        
        return output

# Initializing the model
m = Model(input_size=64)

# Inputs to the model
query = torch.randn(8, 8, 64)
key = torch.randn(8, 8, 64)
value = torch.randn(8, 8, 64)
inv_scale_factor = torch.randn(1)
dropout_p = torch.randn(1)
