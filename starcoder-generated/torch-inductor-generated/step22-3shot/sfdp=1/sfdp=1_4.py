
class Model(torch.nn.Module):
    def __init__(self):
       self.inv_scale_factor = 0.05 

    def forward(self, query, key, value, padding_mask, input_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p, training=self.training)
        output = dropout_qk.matmul(value)
        output = output.masked_fill(padding_mask[None, None, None, :], 0.0)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 4, 8)
key = torch.randn(1, 3, 8, 8)
value = torch.randn(1, 3, 8, 8)
padding_mask = torch.zeros(1, 4, 8, 8).to(torch.bool)
input_mask = torch.zeros(1, 2, 4, 4).to(torch.bool)

