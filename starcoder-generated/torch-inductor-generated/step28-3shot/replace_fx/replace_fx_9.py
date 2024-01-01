
class ModelRandom(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.m = torch.nn.utils.rnn.RandomLayer(1, 1, batch_first=False)
    def forward(self, input: Any) -> Any:
        x = self.m(input)
        x = x.permute(2, 1, 0).contiguous()
        x = x.view(-1, 2)
        return x
# Inputs to the model
torch.randn(1, 3, 3)
