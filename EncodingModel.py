import torch
import torch.nn as nn

class EncodingModel2(nn.Module):
    """
    The neural network class for the visual encoding model.
    """
    AN_NUM_OUTPUT_NEURONS = 4096
    HIDDEN_NEURONS = 2000
    V1_VOXELS = 1294
    V2_VOXELS = 2083
    V3_VOXELS = 1790
    V3A_VOXELS = 484
    V3B_VOXELS = 314
    V4_VOXELS = 1535
    LO_VOXELS = 928
    TOTAL_VOXELS = 8428

    def __init__(self, ROI=None) -> None:
        super().__init__()      # invoke nn.Module

        # Load the pre-trained AlexNet model and freeze all the layers
        AlexNet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.DEFAULT')
        
        # Freeze all but the classifier of AlexNet
        counter = 0
        for child in AlexNet.children():
            if counter < 2:
                for param in child.parameters():
                    param.requires_grad = False
            counter += 1
        
        #for param in AlexNet.parameters():
        #    param.requires_grad = False
            
        # Add the first two children to our model (Note: 'an' abbreviates 'AlexNet' in naming the children of this network)
        self.an_features = list(AlexNet.children())[0]
        self.an_avgpool = list(AlexNet.children())[1]

        # Access all but the last layer of the AlexNet classifier child
        features = list(AlexNet.classifier.children())[:-1]

        # Initalize output layer (by ROI if indicated)
        if ROI == 'V1':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V1_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'V2':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V2_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'V3':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V3_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'V3A':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V3A_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'V3B':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V3B_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'V4':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V4_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        elif ROI == 'LO':
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.LO_VOXELS)])
            self.an_classifier = nn.Sequential(*features)
        else:
            features.extend([nn.Linear(self.AN_NUM_OUTPUT_NEURONS, self.V1_VOXELS)])
            self.an_classifier = nn.Sequential(*features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements foward pass from AlexNet"""
        x = self.an_features(x)
        x = self.an_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.an_classifier(x)

        return x