import vgg
import fista
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

class GALPRUN():
    def __init__(self,batchSize,tea_path):    
        transform_train =transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
        self.vggnet_tea=vgg.vgg16().cuda()
        self.vggnet_tea.load_state_dict(torch.load(tea_path))
        self.vggnet_stu=vgg.vgg16().cuda()
        self.net_D=vgg.net_D().cuda()
        self.optimizer_mask = fista.FISTA(self.vggnet_stu.parameters_mask())
        # self.optimizer_other = optim.SGD(vggnet.parameters_other(), lr=0.001, momentum=0.9)
        # self.optimizer_D = optim.SGD(vggnet.parameters_other(), lr=0.001, momentum=0.9)
        self.optimizer_other = optim.SGD(self.vggnet_stu.parameters_other(), lr=0.01)
        self.optimizer_D = optim.SGD(self.vggnet_stu.parameters_other(), lr=0.01)
        self.other_lr_scheduler = lr_scheduler.StepLR(self.optimizer_other, step_size=30, gamma=0.1)
        self.D_lr_scheduler = lr_scheduler.StepLR(self.optimizer_D, step_size=30, gamma=0.1)
        self.BCELoss = nn.BCELoss().cuda()
        self.MSELoss = nn.MSELoss().cuda()
        self.trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train) 
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batchSize, shuffle=True) 
        self.testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=256, shuffle=True) 
        self.target_real=torch.ones((batchSize,1)).cuda()
        self.target_fake=torch.zeros((batchSize,1)).cuda()
    def train(self,epoch_time):
        for epoch in range(epoch_time):
            self.vggnet_stu.train()
            self.net_D.train()
            D_loss_sum = 0.0
            other_loss_sum = 0.0
            mask_loss_sum = 0.0
            correct = 0.0
            total = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                st=time.time()

                inputs, labels = inputs.cuda(), labels.cuda()
                out_tea=self.vggnet_tea(inputs)

                self.optimizer_mask.zero_grad()
                self.optimizer_other.zero_grad()
                self.optimizer_D.zero_grad()
                out_stu=self.vggnet_stu(inputs)
                out_tea_D=self.net_D(out_tea)
                out_stu_D=self.net_D(out_stu)
                D_loss = self.BCELoss(out_tea_D, self.target_real) + self.BCELoss(out_stu_D, self.target_fake)
                D_loss_sum += D_loss.item()
                D_loss.backward()
                self.optimizer_D.step()
                self.D_lr_scheduler.step()            
                _, predicted = torch.max(out_stu_D.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                
                self.optimizer_mask.zero_grad()
                self.optimizer_other.zero_grad()
                self.optimizer_D.zero_grad()
                out_stu=self.vggnet_stu(inputs)
                out_stu_D=self.net_D(out_stu)
                other_loss = self.BCELoss(out_stu_D, self.target_real)
                l2_regularization = torch.tensor([0],dtype =torch.float32).cuda()
                for param in self.vggnet_stu.parameters_other():
                    l2_regularization += torch.norm(param, 2)/2 #L2 正则化
                other_loss+=l2_regularization
                other_loss+=0.5*self.MSELoss(out_stu_D,out_tea_D)
                other_loss_sum += other_loss.item()
                other_loss.backward()
                self.optimizer_other.step()
                self.other_lr_scheduler.step()
                self.optimizer_mask.step1()
                
                self.optimizer_mask.zero_grad()
                self.optimizer_other.zero_grad()
                self.optimizer_D.zero_grad()
                out_stu=self.vggnet_stu(inputs)
                out_stu_D=self.net_D(out_stu)
                mask_loss = self.BCELoss(out_stu_D, self.target_real)
                mask_loss+=0.5*self.MSELoss(out_stu_D,out_tea_D)  
                mask_loss_sum += mask_loss.item() 
                mask_loss.backward()
                self.optimizer_mask.step2()
            print('[epoch:%d] D_loss: %.03f | other_loss: %.03f | mask_loss: %.03f | Acc: %.3f%% '% (epoch + 1,D_loss_sum / (i + 1),other_loss_sum / (i + 1),mask_loss_sum / (i + 1), 100. * correct / total))
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                for (images, labels) in self.testloader:
                    self.vggnet_stu.eval()
                    images, labels = images.cuda(), labels.cuda()
                    outputs = self.vggnet_stu(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            torch.save(self.vggnet_stu.state_dict(), 'model/vggnet_%03d.pth' % (epoch + 1))
            ed=time.time()
            print("Training Finished, TotalEPOCH=%d,Epochtime=%d" % (epoch,ed-st))

