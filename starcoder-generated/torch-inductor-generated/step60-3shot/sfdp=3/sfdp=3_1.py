
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax = scaled_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax, p=self.dropout_p)
        output = dropout.matmul(value)
        return output

# Initializing the model
m = Model(2, 0.2)

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 512, 256)
value = torch.randn(1, 256, 512)
