
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        self.query = nn.Linear(input_size, num_heads * self.head_size)
        self.key = nn.Linear(input_size, num_heads * self.head_size)
        self.value = nn.Linear(input_size, num_heads * self.head_size)
 
    def forward(self, query, key, value, scale_factor=1, dropout_p=0):
        