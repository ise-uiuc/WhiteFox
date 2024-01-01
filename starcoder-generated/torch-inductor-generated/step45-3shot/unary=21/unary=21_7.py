
class PatternModule(torch.nn.Module):
    def __init__(self):
        super(PatternModule, self).__init__()
        self.conv3x3 = torch.nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.conv1x1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, inputs):
        conv3x3_inputs = self.conv3x3(inputs)
        conv1x1_inputs = self.conv1x1(inputs)
        concat_outputs = torch.cat([conv3x3_inputs, conv1x1_inputs], dim=params.Dim.CHANNEL_DIM)
        return concat_outputs
# Inputs to the model
inputs = torch.randn(2, 32, 64, 64)
