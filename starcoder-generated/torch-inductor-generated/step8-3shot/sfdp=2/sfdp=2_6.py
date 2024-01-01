
class Model(torch.nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        scale_factor = key.size(0) ** 0.25
        self.scale_factor = torch.Tensor([scale_factor])
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout = self.dropout(softmax_qk)
        output = dropout.matmul(value)
        return output

# Initializing the model
query, key, value = torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64)
m = Model(query, key, value)

# Inputs to the model
