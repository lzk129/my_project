from resnet import resnet_50
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader




def detection(path1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型参数
    params = torch.load('my_resnet.pth')

    # 构建新的模型实例
    new_model = resnet_50()

    # 将参数加载到新模型中
    new_model.load_state_dict(params)

    new_model.eval()

    transform = transforms.Compose([
        transforms.Resize(224), #统一图片大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #标准化
    ])
    data = ImageFolder(path1, transform = transform)
    label = 'sit'

    # print(type(data))

    # print(data[0][0][0])

    torch.manual_seed(123)
    data = DataLoader(data, batch_size = 4, shuffle=False, prefetch_factor=2)

    # print(test_dl[0][0][0])

    new_model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(data):
            y_h = new_model(x)
            predict = torch.max(y_h, dim=1)[1]
            predict = predict.numpy().tolist()
            output = predict[0]
            if output == 0:
                label = 'jump'
            elif output == 1:
                label = 'run'
            elif output == 2:
                label = 'sit'
            else:
                label = 'stand'
    return label



