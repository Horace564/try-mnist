# Try MNIST
## Simple implementation of MNIST by PyTorch
- **fc_cpu.py** : implement by 4 fully connected layers, and use CPU resources
- **fc_gpu.py** : implement by 4 fully connected layers, and use GPU resources
- **conv_cpu.py** : implement by the conbination of convolutional layers and fully connected layers, and use CPU resources
- **conv_gpu.py** : implement by the conbination of convolutional layers and fully connected layers, and use GPU resources

## Sample output
- **fc_cpu.py** :   
[epoch 0], accurancy is 11.24%  
[epoch 1], accurancy is 90.32%  
[epoch 2], accurancy is 92.19%  
[epoch 3], accurancy is 93.52%  
[epoch 4], accurancy is 94.45%  
[epoch 5], accurancy is 95.24%  
[epoch 6], accurancy is 95.69%  
[epoch 7], accurancy is 96.16%  
[epoch 8], accurancy is 96.13%  
[epoch 9], accurancy is 96.43%  
[epoch 10], accurancy is 96.54%  
Total training time(CPU) is  21.49 s  

- **fc_gpu.py** :  
[epoch 0], accurancy is 10.32%  
[epoch 1], accurancy is 90.40%  
[epoch 2], accurancy is 92.70%  
[epoch 3], accurancy is 93.88%  
[epoch 4], accurancy is 94.53%  
[epoch 5], accurancy is 95.04%  
[epoch 6], accurancy is 95.34%  
[epoch 7], accurancy is 95.65%  
[epoch 8], accurancy is 95.83%  
[epoch 9], accurancy is 96.15%  
[epoch 10], accurancy is 96.35%  
Total training time(GPU) is  13.09 s  

- **conv_cpu.py** :  
[epoch 0], accurancy is 10.64%  
[epoch 1], accurancy is 96.11%  
[epoch 2], accurancy is 97.75%  
[epoch 3], accurancy is 98.13%  
[epoch 4], accurancy is 98.52%  
[epoch 5], accurancy is 98.59%  
[epoch 6], accurancy is 98.68%  
[epoch 7], accurancy is 98.72%  
[epoch 8], accurancy is 98.83%  
[epoch 9], accurancy is 98.96%  
[epoch 10], accurancy is 98.66%  
Total training time(CPU) is 53.96 s  

- **conv_gpu.py** :  
[epoch 0], accurancy is 10.69%  
[epoch 1], accurancy is 95.12%  
[epoch 2], accurancy is 97.20%  
[epoch 3], accurancy is 97.98%  
[epoch 4], accurancy is 98.23%   
[epoch 5], accurancy is 98.27%   
[epoch 6], accurancy is 98.40%  
[epoch 7], accurancy is 98.55%  
[epoch 8], accurancy is 98.74%  
[epoch 9], accurancy is 98.85%  
[epoch 10], accurancy is 98.83%  
Total training time(CPU) is 15.19s 
