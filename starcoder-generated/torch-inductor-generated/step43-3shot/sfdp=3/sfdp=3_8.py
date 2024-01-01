
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 64, 512)
key = torch.randn(2, 64, 1024)
value = torch.randn(2, 64, 1024)
scale_factor = 1.0 / math.sqrt(512)
