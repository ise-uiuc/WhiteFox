
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.kqv = nn.Linear(input_size, hidden_size * 3)
 
    def forward(self, x1):
        v1 = torch.nn.functional.dropout(torch.nn.functional.glu(nn.functional.gelu(self.kqv(x1))), p=0.1, training=self.training)
        return v1

# Initializing the model
m = Model(32, 64)

# Inputs to the model
x1 = torch.randn(1, 32)
