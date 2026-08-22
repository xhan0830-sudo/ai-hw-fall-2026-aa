import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain range (before normalization)
    # Since MNIST is normalized, we might just clip it to normalized bounds or leave it
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True

    for _ in range(iters):
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(original_images + eta, min=-3.0, max=3.0).detach() # rough norm bounds
        perturbed_images.requires_grad = True

    return perturbed_images

def mifgsm_attack(model, images, labels, epsilon, alpha, iters, decay=1.0):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    momentum = torch.zeros_like(images).detach()

    for _ in range(iters):
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Update momentum
        grad = perturbed_images.grad
        grad_norm = torch.norm(grad, p=1, dim=, keepdim=True)
        grad = grad / (grad_norm + 1e-8)
        momentum = decay * momentum + grad
        
        adv_images = perturbed_images + alpha * momentum.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(original_images + eta, min=-3.0, max=3.0).detach()
        perturbed_images.requires_grad = True

    return perturbed_images
