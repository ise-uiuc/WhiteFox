
class Model(torch.nn.Module):
    def __init__(self):
    super().__init__()
        self.query = torch.nn.Conv2d(128, 2, 1, stride=1, padding=0)
        self.key = torch.nn.Conv2d(128, 2, 3, stride=3, padding=0)
        self.value = torch.nn.Conv2d(128, 2, 3, stride=3, padding=0)
 
    def forward(x1, x2):
        v1 = self.query(x1) # Extract patch embeddings from the query
        v2 = self.key(x2) # Extract patch embeddings from the key
        v3 = self.value(x2) # Extract patch embeddings from the value
        v4 = (v2.transpose(-2, -1) @ v1) / math.sqrt(v1.shape[1]) # Compute the scaled dot product of the query and key
        v5 = v4 + self.masked_attention_weights # Apply the attention mask
        v6 = torch.softmax(v5, dim=-1) # Apply softmax to the output
        v7 = v6 @ v3 # Compute the weighted sum of the value
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 8, 8)
x2 = torch.randn(1, 128, 32, 64)
x3 = torch.randint(0, 2, (1, 64, 16, 16), dtype=torch.bool) # Create a binary attention mask indicating whether the query patch and key patch are both valid. Note that this tensor is already prepared with the batch index and the query patch/key patch dimensions.
