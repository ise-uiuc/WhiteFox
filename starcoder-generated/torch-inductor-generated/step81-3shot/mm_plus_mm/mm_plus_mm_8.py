
class Model(torch.nn.Module):
    def forward(self, weights, inputs):
        h1 = F.relu(torch.mm(weights, inputs), inplace=True)
        h1 = F.relu(h1, inplace=True)
        h1 = F.dropout(h1, training=self.training)
        return h1
# inputs to the model
weights = torch.randn(100,50)
inputs1 = torch.randn(50,24)
