
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3)
y1 = m(x1)
torch.save(m, "model.pt")

# Inputs of all untimed, non-input tensor operations must have shapes that are 1-1 with corresponding shapes of untimed, non-input tensors.

# Input dimensions: (bsz, hidden_dim, seq_len)

