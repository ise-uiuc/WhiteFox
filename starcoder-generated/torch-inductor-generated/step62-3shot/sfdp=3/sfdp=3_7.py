
class Model(torch.nn.Module):
    def __init__(self, input_channels, num_heads, embed_dim, output_channels, dropout_p, bias=True):
        super(GAT, self).__init__()
        
        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.query_network = MultiHeadDotProductAttention(input_channels, input_channels, input_channels, dropout_p, bias=True)

        self.key_network = MultiHeadDotProductAttention(input_channels, input_channels, input_channels, dropout_p, bias=True)

        self.value_network = MultiHeadDotProductAttention(input_channels, input_channels, input_channels, dropout_p, bias=True)

        self.fc = torch.nn.Linear(input_channels, output_channels, bias=bias)

    def forward(self, x1, x2):
        out = self.dropout(torch.cat([
            self.query_network(x1).mean(dim=[-2, -1]),
            self.key_network(x2).mean(dim=[-2, -1]),
            self.value_network(x2).mean(dim=[-2, -1])
        ], dim=1))
        return self.fc(out)

# Initializing the model
m = Model(input_channels=3, num_heads=2, embed_dim=12, output_channels=3, dropout_p=0.2)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
