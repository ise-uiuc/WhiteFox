
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
   
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(0.125)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = torch.tensor(0.2)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
num_batches = 2
query = torch.randn(num_batches, 64, 256, 256)
key = torch.randn(num_batches, 64, 256, 256)
value = torch.randn(num_batches, 64, 256, 256)
