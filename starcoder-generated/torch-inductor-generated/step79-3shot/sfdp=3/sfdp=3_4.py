
class Model(torch.nn.Module):

    def __init__(self, num_queries=20, n_conv_layers=2, n_heads=5, dropout_p=0.05):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.n_heads = n_heads
        self.num_queries = num_queries
        self.conv1 = torch.nn.Conv2d(3, max(n_heads*2, 0), 1, stride=1, padding=1)
        self.layers = [nn.TransformerEncoderLayer(d_model=self.num_queries*int(self.num_queries > 0), num_heads=self.n_heads, dropout=dropout_p, activation='relu',) for _ in range(n_conv_layers)]
        self.model = nn.TransformerEncoder(self.layers, num_layers=n_conv_layers)

    def forward(self, inputs):
        v = self.conv1(inputs)

        if inputs.shape[0] < self.num_queries:
            raise ValueError('The number of input images should be more than or equal to the number of queries.')

        if self.n_heads > 0:
            # (batch_size, channels, height, width) -> (batch_size, num_queries, channels, height, width)
            v = v.repeat(1, self.num_queries, 1, 1, 1).reshape(-1, *v.shape[1:])

        m = {"queries": v[:, :-self.num_queries, :].permute(1, 0, 2),
             "keys": v[:, :self.num_queries, :].permute(1, 0, 2),
             "values": v.permute(1, 0, 2)}

        x = self.model(m['keys'], m['queries'])

        if self.n_heads > 0:
            # (batch_size, num_queries, channels, height, width) -> (batch_size, num_queries, n_features)
            x = x.reshape(-1, self.num_queries, v.shape[1]).permute(1, 0, 2)
        else:
            x = x.permute(1, 0, 2)

        return x

# Initializing the model
m = Model(num_queries=1, n_heads=2)

# Inputs in a shape (batch_size, channels, height, width)
x = torch.randn(16, 3, 64, 64)
