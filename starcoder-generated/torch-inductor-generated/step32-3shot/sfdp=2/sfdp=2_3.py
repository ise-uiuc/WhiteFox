
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self, size, number_heads, projection_size, dropout_p):
        super().__init__()
        self.size = size
        self.number_heads = number_heads
        self.projection_size = projection_size
        self.scale_factor = 1
        if encoder_normalize_before:
            self.scale_factor = self.scale_factor / (self.number_heads * math.sqrt(self.size))
        self.projection = torch.nn.Linear(size, projection_size, bias = False)
        self.dropout = torch.nn.Dropout(dropout_p)
     
    def forward(self, query, key, value, mask = None):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = F.softmax(v2, dim = -1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, value)
        return v5

# Inputs to the model
m = Model(size = 16, number_heads = 8, projection_size = 4, dropout_p = 0.5)
query = torch.randn(1, 32, 16)
key = torch.randn(1, 32, 16)
value = torch.randn(1, 32, 16)
