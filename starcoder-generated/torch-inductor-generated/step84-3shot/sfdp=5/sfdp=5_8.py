
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_seq = 100
    def forward(self, embedding):
        output = embedding.sum(dim=-2)
        output = output[:, 0:80]
        return output
# Inputs to the model
input = torch.randn(1, 50, 1024)
