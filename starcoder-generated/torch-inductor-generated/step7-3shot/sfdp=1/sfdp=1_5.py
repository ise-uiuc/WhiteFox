
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, input_tensor, key, query, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 32, 512, 512)
key = torch.randn(1, 32, 96, 96)
query = torch.randn(1, 32, 96, 96)
value = torch.randn(1, 32, 96, 96)
inv_scale_factor = torch.randn(1, 1)
