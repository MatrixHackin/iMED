import os
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch
import torch.nn as nn


batch=64
size=28
n_epochs=1
learningrate=0.01
latent_dim=100
b1=0.5
b2=0.999
sample_interval=400
cuda=True if torch.cuda.is_available() else False

#准备数据集
os.makedirs("../data/mnist",exist_ok=True)
#初始化DataLoader类（数据集，batch_size,shuffle）
dataloader= DataLoader(
    #数据集一般是一个类,MINIST(root,train,download,transform对数据预处理)
    datasets.MNIST(

        #数据集的根目录
        root="../data/mnist",

        #train用来确定使用的是数据集的训练集还是测试集
        train=True, 

        #download用于在测试集不存在时从网上下载数据集
        download=True,  

        #tansform用来对数据进行预处理
        #利用torchvision.transforms里的compose类对多个预处理进行拼接，通常的处理有：
        # 1.大小,统一大小便于处理和训练（若只传入一个整数则短边调整为整数，长边比例缩放，若两个则变为a*b）
        # 2.张量化（在tensor体系下处理数据）
        # 3.标准化：标准化是把图像数据缩小到【-1，1】或【0，1】内，来提升模型的收敛和训练效果
            #transforms.Normalize的第一个参数为各通道的均值，第二个通道为各通道的标准差
            #通常的像素点有RGB三个通道的灰度值，而MNIST只是黑白图像，所以只有一个通道的值即可
            #Normalize([0.5],[0.5])表示使数据normalize后mean是0.5，std也是0.5
                #这是一种特殊情况？通常情况这里均值和标准差是要自己计算？
        transform=transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch,
    shuffle=True,
)

#定义模型
#我们的模型都是MLP（NN的一种实现方式），所以我们的模型利用父类模型torch.nn中的module来建类
class Generator(nn.Module):
    
    #类还是需要初始化函数来开头，初始化函数传入参数self是约定俗称的将类实例化为self，接下来的代码中便可以用self来访问类的属性和方法
    def __init__(self):
        #这里我们是利用继承父类建类,子类和父类本身都要传入super()函数进行实例化
        super(Generator,self).__init__()

        #既然我们想要generator是一个MLP，接下来我们便来构建他的各个层
        #我们把一个可学习的线性映射层，一个batch normalization层，以及一个激活层来共同组成一个block(3个层组成的list)
        def block(in_feat, out_feat, normalize=True):

            #layers是一个有序的结构，所以我们使用list这个数据结构
            # 首先是一个可学习的线性映射nn.Linear来改变特征空间的维度，每个特征空间的维度会代表图像的某一特征
            layers=[nn.Linear(in_feat, out_feat)]

            #然后是一个可以选择使用的batch normalization层，第一个参数是待归一化数据的维度，第二个参数是用于计算均值和方差的栋梁参数，这一参数控制了训练过程中对均值和方差的更新的速度，通常设置为接近1的值
            #在训练过程中，随着参数的改变导致每一层输入数据的分布总在变化，我们把这种变化称为内部协变量变化，会导致可能出现梯度消失或者梯度爆炸的情况而导致训练效果不稳定，因此我们需要一个手段来避免这种情况的发生，所以我们对每一层的输入进行归一化处理
            #batchNormalization这一算法计算得到该批数据的均值和方差后进行归一化处理使数据分布均值趋近于0，方差趋近于1
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            
            #最后是一个激活层，用于使训练过程对非线性模型的拟合效果更好，
            # #此处使用leaKyReLU函数：f(x) = max(ax, x)，这是在传统ReLU函数上做到的优化，ReLU函数在接受负数输入的时候会映射为0，这可能导致在训练过程中出现梯度消失的问题，于是我们使负数映射为一个很小的值，来避免这种情况
            #第一个参数是LeakyReLU的泄露率，较小的泄露率可以使负数部分仍映射有较小的斜率，较大的泄露率可能会导致激活函数的非线性性质消失而影响激活效果
            #第二个参数inplace=True是为了节省空间而直接对输入数据进行激活映射，但这样会丢失掉映射前的数据，此处映射前的数据没用，所以我们就这么干
            layers.append(nn.LeakyReLU(0.2,inplace=True))

            return layers
        
        #定义模型是一系列神经网络层和模块的序列
        self.model=nn.Sequential(
            #*表示将block的返回值解包为层添加到Sequential的参数列表
            #将潜在特征空间逐层加倍，可以增加模型的复杂度和表示能力，以适应更复杂的模式和特征。通过每次将层的大小乘以2，模型可以逐渐学习到更抽象和复杂的特征，从而提高其表征能力。
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            #最终将特征空间维度映射为表示为图片需要的维度即1*28*28,利用np.prod对元组中的元素取积
            nn.Linear(1024,int(np.prod((1,28,28)))),
            #最后通过双曲正切函数转换将数据值都压缩在【-1，1】之间
            nn.Tanh()
        )
    
    def forward(self,z):
        #利用传入的随机噪音z在模型中生成一个img向量
        img=self.model(z)
        #再利用img.view调整张量大小为（1*1*28*28）?
        img=img.view(img.size(0))
        print(img.shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self) :
        super(Discriminator,self).__init__

        self.model=nn.Sequential(
            #通过一系列层把向量变成1维的
            nn.Linear(int(np.prod((1,28,28))),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )

    def forward(self,img):
        #首先用view把图像向量变成二维的以便传入模型进行计算
        img_flat=img.view(img.size(0),-1)
        validity=self.model(img_flat)
        return validity
    
#定义损失函数
adversarial_loss=torch.nn.BCELoss()

#实例化生成器和辨别器
generator=Generator()
discriminator=Discriminator()

#用nn中的cuda()为利用nn构建的模型设置cuda运算
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

#Optimizers用于计算梯度并更新参数（optimizer.step()），传入模型参数，lr,betas(betas在Adam优化中用于计算梯度，具体再看)来进行
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learningrate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters,lr=learningrate,betas=(b1,b2))

#规定tensor用cpu还是gpu进行计算
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for epoch in range(n_epochs):
    #遍历规定的epochs数
    #enumerate用于将可遍历的对象同时列出遍历下标和数据,第一个数据给了i,后面的数据给了元组（imgs,_）_是占位符
    #由于dataloader分了batchsize是64，所以每个tensor都是64张图片，每张图片只有1个通道，共28*28个像素点记录灰度值，是（64*1*28*28）的张量
    for i,(imgs,_) in enumerate(dataloader):

        # -----------------
        #  Train Generator
        # -----------------

        #在计算梯度并更新参数前要先将之前累积的梯度清零
        optimizer_G.zero_grad()

        # Sample noise as generator input
        #Variable(Tensor())类可以封装一个变量为tensor使其能够进行tensor的运算,这是产生一个64张图片的随机噪声,即一个数据取自标准正态分布的64*latent_dim张量
        #latent space是数据的潜在空间，我们每张图片的数据其实是1*28*28的3维数组，我们现在把这些数据压缩到一个latent_dim维的latent space里面，每一个维度的数据都代表着数据的一个特征
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        
        # Generate a batch of images
        #利用一个batch(64张图片)的随机噪声生成gen_imgs
        gen_imgs = generator(z)

        #计算生成的图片被discriminator辨认的情况和真实情况（辨认完全发现是对的显示1）的Loss
        #首先要建立一个同样大小的全1张量用来表示辨别为真，一个全0的张量表示鉴别为假，用他们来计算loss,这些张量不需要求梯度，所以添加一个不用求梯度的标签来优化
        valid=Variable(Tensor(imgs.size(0),1).fill_(1.0),requires_grad=False)
        fake=Variable(Tensor(imgs.size(0),1).fill_(0.0),requires_grad=False)
        g_loss = adversarial_loss(discriminator(gen_imgs),valid)

        #将g_loss返回给模型进行调参
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        #discriminator的loss定义为(他辨别真实图片为真的loss+他辨别生成图片为假的loss)/2
        #真实图片同样需要张量化
        real_imgs=Variable(imgs.type(Tensor))

        real_loss=adversarial_loss(discriminator(real_imgs),valid)
        #detach()用于舍弃掉？求梯度功能？
        fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake)
        d_loss=(real_loss+fake_loss)/2

        #传回loss并调参
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D_loss: %f] [G_loss: %f]"
            #using %() to formatly print
            #.item() is used to extract the scalar value of the loss from a tensor
            %(epoch,n_epochs,i,len(dataloader),d_loss.item(),g_loss.item())
        )

        #len(dataloader) get the number of batches of data loader
        batches_done=epoch*len(dataloader)+i  
        #记录一共训练完了几个batch,每训练完sample_interval个batch输出一个图片看看效果
        if batches_done%sample_interval==0:
            #save_image(张量数据，路径，输出行数，是否归一化)
            save_image(gen_imgs.data[:25],"images/%d"%batches_done,nrow=5,normalize=True)



        

        
