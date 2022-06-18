from mup import get_shapes, make_base_shapes
from ConvNet_architectures.resnet import ResNet18, ResNet34, ResNet50

base_model=ResNet50(num_classes=100, wm=1)
delta_shape=ResNet50(num_classes=100, wm=8)
shape_base=make_base_shapes(base_model,delta_shape, savefile='./base_shapes/resnet50.bsh')