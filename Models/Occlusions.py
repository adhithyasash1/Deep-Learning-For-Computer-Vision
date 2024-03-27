import torch
import torchvision.models as models
from torch.autograd import Variable

# Load pre-trained VGG16
model = models.vgg16(pretrained=True)

# Preprocess your input image and convert it to a torch variable
# Assuming preprocess_image function and img variable are defined
X = preprocess_image(img)  # img is your input image
X.requires_grad_()

# Forward pass
output = model(X)
output_max_index = output.argmax()
output_max = output[0, output_max_index]

# Backward pass to get gradients
output_max.backward()

# Saliency is the absolute value of the gradient
saliency, _ = torch.max(X.grad.data.abs(), dim=1)


def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    width, height = image.shape[-2], image.shape[-1]
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
    heatmap = torch.zeros((output_height, output_width))

    for h in range(0, height):
        for w in range(0, width):
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]
            heatmap[h, w] = prob

    return heatmap
