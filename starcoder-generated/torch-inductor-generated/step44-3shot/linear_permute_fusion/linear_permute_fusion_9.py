
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x2):
        v2 = torch.mm(x2, self.linear.weight)
        v3 = v2.permute(2, 0, 1)
        v4 = v3.contiguous()
        return v4
# Inputs to the model
# x2 = torch.randn(3, 2) # (seq_length, batch_size, num_hidden)
x2 = torch.randn(2, 3) # (batch_size, seq_length, num_hidden)
