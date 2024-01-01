 inputs
self.output_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
key = output[:, 0, :].unsqueeze(1)
query = x[:, 0, :].unsqueeze(1)
value = x
inv_scale_factor = math.sqrt(query.size(-1))

# Description of model
x = torch.randn(1, 16, 256)
x = self.output_projection(x)
query = x[:, 0, :].unsqueeze(1)
key = x
value = x

