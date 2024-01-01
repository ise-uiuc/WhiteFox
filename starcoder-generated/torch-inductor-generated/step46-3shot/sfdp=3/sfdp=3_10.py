
class Model(torch.nn.Module):
    def __init__(self, input_length, hidden_size, num_heads, num_layers):
        super().__init__()
        self.transformer = torch.nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
 
    def forward(self, x1, x2):
        x3 = self.transformer(x1, x2)
        return x3
 
# Initializing the model
num_heads = 2
hidden_size = 8
num_layers = 2
input_length = 96
m = Model(input_length, hidden_size, num_heads, num_layers)
 
# Inputs to the model
x1 = torch.rand(1, 96, 64).to(device=0)
x2 = torch.rand(1, 96, 64).to(device=0)
