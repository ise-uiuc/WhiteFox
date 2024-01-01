
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1 / (10.0 ** 0.5), dropout_p=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = query.matmul(key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        return scaled_qk.softmax(dim=-1) * torch.nn.functional.dropout(scaled_qk, p=self.dropout_p).matmul(value)
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 6, 20)
key = torch.randn(1, 8, 20)
value = torch.randn(1, 8, 16)
