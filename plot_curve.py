import matplotlib.pyplot as plt

# Initialize lists to store epoch and loss values
epochs = []
losses = []

# Open and read the log file
with open('log.txt', 'r') as file:
    lines = file.readlines()
# print("lines", lines)
# Iterate through each line
for line in lines:
    if "\'loss\'" not in line:
        continue
    # Remove leading and trailing whitespace and curly braces
    line = line.strip('{').strip()
    line = line[:-1]
    print("line", line)
    
    # Split the line based on commas and colons
    items = line.split(', ')
    
    # Extract loss and epoch values
    loss = float(items[0].split(':')[1])
    epoch = float(items[2].split(':')[1])
    
    # Append values to the lists
    epochs.append(epoch)
    losses.append(loss)

# Plot the loss based on epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-')
print(epochs)
print(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.grid(True)
plt.savefig('loss_vs_epoch.png')
