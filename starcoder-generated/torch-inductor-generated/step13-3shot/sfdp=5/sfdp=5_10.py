
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 512
    def forward(self, x):
        x = x + x
        output = F.relu(x)
        return output
# Inputs to the model
x = torch.randn(1, self.seq_len, 512)
