
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value, dropout_p, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / self.inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(inv_scale_factor=50.)

# Inputs to the model
query = torch.randn(1, 1, 50)
key = torch.randn(1, 100, 50)
value = torch.randn(1, 100, 10)
dropout_p = 0.1
input_mask = torch.zeros(query.size(0), key.size(-2)).byte() # Use the built-in function zeros() to generate a tensor with the same shape as query and the type Byte. Set the mask elements to 1.
