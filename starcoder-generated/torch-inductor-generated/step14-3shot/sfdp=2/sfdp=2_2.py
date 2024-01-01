
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, _, scale_factor, dropout_p):
        qk = torch.matmul(query, _.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(_)
        return output[0]

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 128)
key = torch.randn(1, 64, 256) # key can have different sequence length than the query for different batches
value = torch.randn(1, 64, 256)
