
nn.Linear

# Inputs to the model
t1 = torch.randn(1, 224, 224, 3) # The original unsliced tensor along dimension 3
t2 = torch.randn(1, 225, 225, 32) # A sliced tensor along dimension 3
x = torch.cat([t1, t2], dim=3) # A concatenated tensor along dimension 3
y = x[:, :, :, 0:224] # The original concatenated tensor along dimension 3
z = x[:, :, :, 0:225] # The sliced tensor along dimension 3

