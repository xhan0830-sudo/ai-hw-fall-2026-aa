import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net
from attacks import fgsm_attack, pgd_attack, mifgsm_attack

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    correct_clean = 0
    success_fgsm = 0
    success_pgd = 0
    success_mifgsm = 0
    total = 0
    
    epsilon = 0.3
    alpha = 0.01
    iters = 40

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        total += 1

        # --- Clean Eval ---
        output = model(data)
        init_pred = output.max(1, keepdim=True)
        if init_pred.item() == target.item():
            correct_clean += 1
        else:
            continue # Only attack images that were correctly classified

        # Calculate grad for FGSM
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # --- FGSM ---
        perturbed_data_fgsm = fgsm_attack(data, epsilon, data_grad)
        output_fgsm = model(perturbed_data_fgsm)
        if output_fgsm.max(1, keepdim=True).item() != target.item():
            success_fgsm += 1

        # --- PGD ---
        perturbed_data_pgd = pgd_attack(model, data, target, epsilon, alpha, iters)
        output_pgd = model(perturbed_data_pgd)
        if output_pgd.max(1, keepdim=True).item() != target.item():
            success_pgd += 1

        # --- MI-FGSM ---
        perturbed_data_mifgsm = mifgsm_attack(model, data, target, epsilon, alpha, iters)
        output_mifgsm = model(perturbed_data_mifgsm)
        if output_mifgsm.max(1, keepdim=True).item() != target.item():
            success_mifgsm += 1
            
        if total >= 1000: # Stop at 1000 for faster testing
            break

    print(f"Total Test Samples: {total}")
    print(f"Clean Recognition Rate: {correct_clean / total * 100:.2f}%")
    print(f"FGSM Attack Success Rate (ASR): {success_fgsm / correct_clean * 100:.2f}%")
    print(f"PGD Attack Success Rate (ASR): {success_pgd / correct_clean * 100:.2f}%")
    print(f"MI-FGSM Attack Success Rate (ASR): {success_mifgsm / correct_clean * 100:.2f}%")

if __name__ == '__main__':
    test()
