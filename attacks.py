import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon):
    """Fast Gradient Sign Method (FGSM)"""
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # Add perturbation gradient sign
    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=40):
    """Iterative FGSM / Projected Gradient Descent (PGD)"""
    original_images = images.clone().detach()
    adv_images = images.clone().detach()
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Step in direction of gradient sign
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        
        # Project back into epsilon ball and clip to valid image dynamic range [0, 1]
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, min=0, max=1)
        
    return adv_images.detach()


def momentum_ifgsm_attack(model, images, labels, epsilon, alpha=0.01, iters=40, decay=1.0):
    """Momentum Iterative FGSM (MI-FGSM)"""
    original_images = images.clone().detach()
    adv_images = images.clone().detach()
    momentum = torch.zeros_like(images)
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Calculate L1 normalized gradient
        grad = adv_images.grad
        grad_flat = grad.view(grad.shape[0], -1)
        l1_norm = torch.norm(grad_flat, p=1, dim=1).view(-1, 1, 1, 1) + 1e-10
        normalized_grad = grad / l1_norm
        
        # Update momentum vector
        momentum = decay * momentum + normalized_grad
        
        # Update adversarial image
        adv_images = adv_images.detach() + alpha * momentum.sign()
        
        # Project back into epsilon ball and clip to valid image range [0, 1]
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, min=0, max=1)
        
    return adv_images.detach()
