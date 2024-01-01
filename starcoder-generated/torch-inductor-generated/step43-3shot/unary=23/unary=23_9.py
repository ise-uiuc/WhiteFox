
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 7, 5, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = cv.dnn.blobFromImage(input_image, scalefactor=1.0 / 255.0, size=(image_width, image_height), mean=(0.485, 0.456, 0.406), swapRB=True, crop=False)
