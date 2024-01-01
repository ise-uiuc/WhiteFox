
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.9
    
    def forward(self, query, key, value, scale_factor=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if scale_factor:
            qk = qk * scale_factor.unsqueeze(dim=-1).unsqueeze(dim=-1)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
dropout_p = 0.9
scale_factor = torch.rand(1, 10, 10)

# Inputs to the model
query = torch.randn(1, 10, 50)
key = torch.randn(1, 10, 40)
value = torch.randn(1, 10, 40)
