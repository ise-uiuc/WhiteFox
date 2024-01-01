
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(8)

# Inputs to the model
query = torch.randn(64, 8, 20, 32)
key = torch.randn(64, 8, 20, 32)
value = torch.randn(64, 8, 20, 32)
inv_scale_factor = 0.5 * torch.randn(64, 8, 1, 1)
dropout_p = 0.1
