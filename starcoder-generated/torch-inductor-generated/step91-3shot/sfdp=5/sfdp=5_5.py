
class Model(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_classes=10, num_heads=12, max_len=1000):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_len = max_len
        self.attn = nn.MultiheadAttention(input_dim, num_heads)
        self.fc = nn.Linear(self.hidden_dim * self.num_heads, self.num_classes)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        attn_output = attn_output.flatten(1)
        return self.fc(attn_output)
# Inputs to the model
x = torch.randn(1, self.max_len, self.hidden_dim)
