
class Model(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_size: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.projection = torch.nn.Linear(input_channels, hidden_size)
        self.activation = torch.nn.ReLU()
 
    def forward(self, x1, x2):
        v1 = self.projection(x1)
        v2 = v1.transpose(-2, -1)
        v3 = torch.matmul(x2, v2)
        v4 = v3.div(self.num_heads ** (-0.5))
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=self.dropout_p)
        v7 = torch.matmul(v6, x1)
        return v7

# Initializing the model
m = Model(8, 4, 8, 0.1)

# Inputs to the model
x1 = torch.randn(1, 4, 8)
x2 = torch.randn(1, 5, 4)
