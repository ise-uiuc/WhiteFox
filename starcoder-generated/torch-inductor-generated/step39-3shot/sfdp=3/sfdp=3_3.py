
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0]))
        self.dropout_p = torch.nn.Parameter(torch.tensor([0.0]))
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 128)
key = torch.randn(1, 2, 128)
value = torch.randn(1, 2, 128)
