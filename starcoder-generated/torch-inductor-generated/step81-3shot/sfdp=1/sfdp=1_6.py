
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()
 
# Inputs to the model
q = torch.randn(2, 8, 20) # Input tensor for the query. Here the rank of the tensor is (2, 8, 20), where 2 is the batch size, 8 and 20 are the hidden dimension and the sequence length, respectively.
k = torch.randn(2, 20, 32) # Input tensor for the key. Here the rank of the tensor is (2, 20, 32), where 2 is the batch size, 20 and 32 are the hidden dimension and the sequence length, respectively.
v = torch.randn(2, 8, 32) # Input tensor for the value. Here the rank of the tensor is (2, 8, 32), where 2 is the batch size, 8 and 32 are the hidden dimension and the sequence length, respectively.
inv_scale_factor = float(1.0 / 64) # The inverse scale factor (64)
