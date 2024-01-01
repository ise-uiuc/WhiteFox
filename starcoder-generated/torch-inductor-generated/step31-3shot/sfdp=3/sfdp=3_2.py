
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0/(head_size**0.5)
        self.dropout_p = 0.2
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
input_tensor = torch.randn(1, 8, 64, 64)
m = Model(input_tensor=input_tensor)

# Inputs to the model
query = torch.randn(1, 8, 32, 64)
key = torch.randn(1, 8, 64, 32)
value = torch.randn(1, 8, 32, 64)
