
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        output = torch.matmul(query, key.transpose(-2, -1))
        scaled_output = output.div(inv_scale_factor)
        softmax_output = scaled_output.softmax(dim=-1)
        if training:
            dropout_output = torch.nn.functional.dropout(softmax_output, p=dropout_p)
        else:
            dropout_output = softmax_output
        output = dropout_output.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 512)
key = torch.randn(1, 12, 512)
value = torch.randn(1, 12, 512)
