
class Model(torch.nn.Module):
    def __init__(self):
         super().__init__()
 
    def forward(self,query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initialize the model for testing
m = Model()

# Inputs to the model
query = torch.randn(1,16,784)
key = torch.randn(1,16,784)
value = torch.randn(1,16,784)
inv_scale_factor = torch.randint(1,10,(1,))
dropout_p = random.uniform(0,1)
