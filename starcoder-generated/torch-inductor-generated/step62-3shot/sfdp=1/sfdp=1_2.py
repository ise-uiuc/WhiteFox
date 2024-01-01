
class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inv_scale_factor = self.args.key_length ** -0.5
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.args.dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(args)

# Inputs to the model
query = torch.randn(1, 64, args.query_length, args.key_length) # Query tensor
key = torch.randn(1, 64, args.key_length) # Key tensor
value = torch.randn(1, 64, args.value_length, args.key_length) # Value tensor
