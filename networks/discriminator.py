from torch import nn

class get_discriminator(nn.Module):
    def __init__(self, img_ch=2):
        super(get_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 64, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), #1
    )
        

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
        return x

class get_dtwo_discriminator(nn.Module):
    def __init__(self, img_ch=5):
        super(get_dtwo_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 64, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), #1
    )


    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
        return x




class get_done_entropy_discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super(get_done_entropy_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 16, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(16),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


class get_exit_discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super(get_exit_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 64, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


