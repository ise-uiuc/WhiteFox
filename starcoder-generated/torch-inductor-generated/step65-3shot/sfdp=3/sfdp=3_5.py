
class Model(torch.nn.Module):
    def __init__(self, dropout: float, scale_factor: float):
        super().__init__()
        self.dropout = dropout
        self.scale_factor = scale_factor
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout=0.1, scale_factor=10)

# Inputs to the model
query = torch.randn(1, 8, 16)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
