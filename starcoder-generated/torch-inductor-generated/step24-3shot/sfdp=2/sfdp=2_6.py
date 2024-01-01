
class Model(torch.nn.Module):
    def __init__(self, query_channels, key_channels, value_channels, n_heads, input_resolution, dropout_p):
        super().__init__()
        self.query_conv = torch.nn.Conv2d(
                query_channels, query_channels, kernel_size=1, stride=1, padding=0
            )
        
        self.key_conv = torch.nn.Conv2d(
                key_channels, key_channels, kernel_size=1, stride=1, padding=0
            )
        
        self.value_conv = torch.nn.Conv2d(
                value_channels, value_channels, kernel_size=1, stride=1, padding=0
            )
        
        self.n_heads = n_heads
        self.input_resolution = input_resolution

        # Create the positional encoding
        self.positional_encoding = torch.nn.Parameter(
            torch.randn(
                input_resolution[0] * input_resolution[1],
                positional_encoding_channels // n_heads,
                n_heads,
            )
        )
        
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Get the query, key, and value
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        # Reshape the query, key, value, and positional encoding
        q = q.reshape(batch_size, self.n_heads, channels // self.n_heads, height, width)
        k = k.reshape(batch_size, self.n_heads, channels // self.n_heads, height, width)
        v = v.reshape(batch_size, self.n_heads, channels // self.n_heads, height, width)
        
        q = q.permute(2, 0, 1, 3, 4)
        k = k.permute(2, 0, 1, 3, 4)
        v = v.permute(2, 0, 1, 3, 4)
        
        q = q.reshape(channels // self.n_heads, batch_size * self.n_heads, height * width)
        k = k.reshape(channels // self.n_heads, batch_size * self.n_heads, height * width)
        v = v.reshape(channels // self.n_heads, batch_size * self.n_heads, height * width)

        # Compute the position encoding and add it to the query
        pe = torch.nn.functional.embedding(
            torch.arange(height * width), self.positional_encoding
        )
        q += pe
        q = q.reshape(batch_size * self.n_heads, channels // self.n_heads, height, width)
        
        # Perform dot product
        attn = q.matmul(k.transpose(0, 1))
        attn = attn.reshape(batch_size, self.n_heads, height * width, height * width)
        
        # Scale the dot product
        attn = attn.scale(1.0 / (channels ** (1.0 / 4.0)))
        
        # Apply softmax to the dot product
        attn = attn.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        attn = self.dropout(attn)
        
        # Compute the dot product of the dropout output and the value
        attn = attn.matmul(v)
        
        # Reshape and transpose the dot product of the dropout output and the value
        attn = attn.reshape(batch_size, self.n_heads, channels // self.n_heads, height, width)
        attn = attn.permute(0, 2, 1, 3, 4)
        
        # Reshape the dot product of the dropout output and the value
        out = attn.reshape(batch_size, channels, height, width)
        
        return out, attn

# Initializing the model
n_heads = 2
query_channels = 32
key_channels = 32
value_channels = 32
positional_encoding_channels = 32

m = Model(query_channels, key_channels, value_channels, n_heads, (64, 64), 0.0)

# Inputs to the model
x = torch.randn(1, query_channels, 64, 64)
__output_1__, __output_2__ = m(x)

