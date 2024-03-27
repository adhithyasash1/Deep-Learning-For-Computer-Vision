import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Assuming model is your PyTorch model and input_tensor is your input data
model.eval()  # Set the model to evaluation mode

# Initialize Integrated Gradients
integrated_gradients = IntegratedGradients(model)

# Compute attributions using Integrated Gradients
attributions_ig = integrated_gradients.attribute(input_tensor, target=label_index)

# Visualize the attributions
viz.visualize_image_attr(attributions_ig[0].cpu().detach().numpy(), 
                         input_tensor.cpu().detach().numpy(), 
                         method='heat_map', 
                         show_colorbar=True, 
                         sign='positive',  # Shows the positive attribution
                         title='Integrated Gradients Attribution')
