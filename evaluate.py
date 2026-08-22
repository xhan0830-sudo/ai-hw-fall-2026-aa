import torch
from torchvision import datasets, transforms
from train import CNNMNIST
from attacks import fgsm_attack, pgd_attack, momentum_ifgsm_attack

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    model = CNNMNIST().to(device)
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()

    # 1. Clean Recognition Rate
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    clean_acc = (correct / total) * 100
    print(f"Clean Model Recognition Rate: {clean_acc:.2f}%\n")

    # 2. Evaluate Attacks across Epsilon values
    epsilons = [0.1, 0.2, 0.3]
    attacks = {
        "FGSM": fgsm_attack,
        "I-FGSM / PGD": lambda m, i, l, e: pgd_attack(m, i, l, e, alpha=0.01, iters=40),
        "Momentum I-FGSM": lambda m, i, l, e: momentum_ifgsm_attack(m, i, l, e, alpha=0.01, iters=40, decay=1.0)
    }

    for eps in epsilons:
        print(f"--- Testing Epsilon = {eps} ---")
        for name, attack_fn in attacks.items():
            successful_attacks = 0
            correctly_classified_initially = 0
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Evaluate clean predictions first to compute true ASR
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                mask = (preds == labels)
                
                if mask.sum() == 0:
                    continue
                
                # Generate adversarial images only for initially correctly classified samples
                clean_imgs = images[mask]
                clean_lbls = labels[mask]
                correctly_classified_initially += clean_imgs.size(0)
                
                adv_imgs = attack_fn(model, clean_imgs, clean_lbls, eps)
                adv_outputs = model(adv_imgs)
                _, adv_preds = torch.max(adv_outputs, 1)
                
                # Attack is successful when model misclassifies
                successful_attacks += (adv_preds != clean_lbls).sum().item()
                
            asr = (successful_attacks / correctly_classified_initially) * 100
            print(f"[{name}] Attack Success Rate (ASR): {asr:.2f}%")
        print()

if __name__ == "__main__":
    evaluate()
