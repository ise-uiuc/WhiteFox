
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, drop_out_p):
        scale_factor = key.size(-2)**-0.5
        scaled_dot = torch.matmul(query * scale_factor, key.transpose(-2, -1))
        softmax_output = torch.nn.functional.softmax(scaled_dot, dim=-1)
        drop_out_output = torch.nn.functional.dropout(softmax_output, p=drop_out_p)
        output = drop_out_output.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
drop_out_p = 0.5
