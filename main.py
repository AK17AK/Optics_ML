import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

# Parameters
n_fix = 28 * 28  # Size of input image (28x28)
n = 1300  # Number of weights
batch_size = 1
num_classes = 10
learning_rate = 0.00001
num_epochs = 10

# Define the directory where the dataset will be stored
data_dir = './mnist_data'

# Define transformations (normalizing the data)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Download the dataset
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

# Wrap in DataLoader for easy iteration
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def system_init(num, n_fix):
    size = (num, num)
    # Generate random phases between 0 and 2Ï€
    phases = torch.rand(size) * 2 * torch.pi
    # Create unitary matrix using QR decomposition
    matrix = torch.cos(phases) + 1j * torch.sin(phases)
    matrix, _ = torch.linalg.qr(matrix, mode='reduced')
    x = torch.rand((num - n_fix, 1))
    return matrix, x


sys, w_0 = system_init(n, n_fix)


def fiber_func(input_with_weights, target, system):
    # Reshape the first 1300 elements
    input_data = input_with_weights[:1300].to(system.dtype)

    v_out = system @ input_data
    reshaped = v_out.view(-1, 130)  # Shape: (10, 130)

    # Apply softmax along each chunk of 130 elements and sum
    summed = reshaped.sum(dim=1)
    abs_summed = torch.abs(summed)
    out = F.softmax(abs_summed, dim=0)  # Shape: (10,)
    l_out = out.argmax()
    # Compute cross-entropy loss
    target = target.view(-1).long()
    loss = F.cross_entropy(out.unsqueeze(0), target)
    return loss, l_out


def numerical_gradient_parallel(system, image, w, target, epsilon=1e-5, device="cuda"):
    # Move data to device
    image = image.to(device)
    w = w.to(device)
    target = target.to(device)
    system = system.to(device)

    combined_input = torch.cat([image.flatten(), w.flatten()]).clone().detach()
    # Number of weights
    num_weights = w.numel()
    # Create perturbed inputs in parallel
    perturbation = torch.eye(num_weights, device=device) * epsilon  # Shape: (num_weights, num_weights)

    zeros_tensor = torch.zeros_like(image.flatten(), device=device).unsqueeze(0)  # Shape: (1, n_fix)
    zeros_tensor_repeated = zeros_tensor.repeat(num_weights, 1)  # Shape: (num_weights, n_fix)

    pos = torch.cat([zeros_tensor_repeated, perturbation], dim=1) + combined_input.repeat(num_weights, 1)
    neg = combined_input.repeat(num_weights, 1)

    # Compute losses in parallel
    losses_pos = torch.tensor([fiber_func(pos[i], target, system)[0] for i in range(pos.size(0))], device=device)
    losses_neg = torch.tensor([fiber_func(neg[i], target, system)[0] for i in range(neg.size(0))], device=device)

    # Compute gradients
    gradients = (losses_pos - losses_neg) / (epsilon)
    return gradients.view_as(w), losses_neg[0].to(torch.int)

def train(system, w, train_loader, n, n_fix, learning_rate):
    # Define the training loop
    x = w
    x =x.to(device)
    cumulative_loss = 0.0
    cumulative_grad =torch.zeros(n - n_fix).unsqueeze(1)
    cumulative_grad = cumulative_grad.to(device)
    j = 0
    for epoch in range(num_epochs):
        # Iterate through batches of training data
        for i, (images, labels) in enumerate(train_loader):
            j+=1
            if (j == 10000):
                break
            images, labels = images.to(device), labels.to(device)
            grad = torch.zeros(n - n_fix)
            grad , loss = numerical_gradient_parallel(system, images, x, labels, epsilon=1e-5, device="cuda")
            cumulative_loss += loss
            #print(f"grad shape: {grad.shape},cumulative_grad shape: {cumulative_grad.shape}")

            cumulative_grad+= grad
            if i % 100 == 99:  # Print every 100 batches
                x = x - learning_rate * (cumulative_grad/100)
                cumulative_grad = torch.zeros(n - n_fix).unsqueeze(1)
                cumulative_grad = cumulative_grad.to(device)
                print((cumulative_loss/100))
                cumulative_loss = 0.0
        #test
        success = 0
        for i, (images, labels) in enumerate(test_loader):
            image = images.to(device)
            w = w.to(device)
            labels = labels.to(device)
            system = system.to(device)

            input_with_weights = torch.cat([image.flatten(), w.flatten()]).detach()
            _,label_out = fiber_func(input_with_weights,labels, system)
            if(label_out==labels):
                success +=1
        print((success/10000))




train(sys, w_0, train_loader, n, n_fix, learning_rate)
