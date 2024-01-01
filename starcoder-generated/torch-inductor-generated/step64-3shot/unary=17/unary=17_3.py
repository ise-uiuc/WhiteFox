
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_same_padding = torch.nn.Conv2d(32, 128, (3, 3), strides=(2, 2), padding='same')
    def forward(self, inputs)
        v1 = self.conv2d_same_padding(inputs)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
inputs = tf.placeholder(tf.float32, shape=(10, 32, 299, 299))
