
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size=64, activation=nn.ReLU):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.attention = scaled_dot_product_attention(hidden_size)
        self.dropout_layer = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = activation
 
    def forward(self, x):
        x = rearrange(x, 'b n (h d) -> b h n d', n=self.num_heads, h=self.input_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        output = self.attention(query, key, value)
        output = rearrange(output, 'b h t d -> b t (h d)')
        output = self.dropout_layer(output)
        return self.fc(output)

# Initializing the model
m = Model(2048, 1024, 48)

# Inputs to the model
x = torch.randn(16, 30, 2048)
