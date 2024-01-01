
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(8, 8)
        self.query = torch.nn.Linear(8, 8)
 
    def forward(self, input_tensor, attention_mask):
        attention_mask.data = attention_mask.data * -10000.0
        q = self.query(input_tensor) @ self.key(input_tensor).transpose(-2, -1) / math.sqrt(self.query(input_tensor).size(-1))
        q = q + attention_mask
        attn_weight = torch.softmax(q, dim=-1)
  
        output = attn_weight @ input_tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(64, 8, 56, 56)
attention_mask = attention_mask = torch.randn(64, 1, 56, 56)
