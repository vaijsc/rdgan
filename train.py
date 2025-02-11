a = 'sudo rm ./saved_info/rdgan/cifar10/bs64/50*.jpg'
for i in range(50):
    a += f'\nsudo rm ./saved_info/rdgan/cifar10/bs64/{50 - i}*.jpg'
print(a)