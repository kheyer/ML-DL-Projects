from .imports import *
from .torch_imports import *
from .core import *
from .dataset import open_image
from matplotlib import rcParams, animation, rc
from ipywidgets import interact, interactive, fixed
from ipywidgets.widgets import *

def plot_cnn_visuals(PATH, filename, model, transformer=None, dpi=None,
total_layers=None, composite=True, individuals=True, animation=True,
composite_figsize=(96,96), animation_figsize=(12,12), resize=(256,256),
resize_animation=(256,256), animation_interval=200, folder_suffix=1,
return_acts=True):
    """
    Takes a trained convolutional model and the filename of an input image
    to generate images of network activations

    Arguments:
        PATH : file path to directory where ouputs will be saved

        filename : filename of input image along PATH

        model : trained convolutional model (e.g. learn.model)

        transformer : function to normalize and resize data as the model to what the
            model was trained on
                e.g. the val_tfms function from
                trn_tfms, val_tfms = tfms_from_model(arch, sz)
            If no function is given, the input image will be resized to a square and
            normalized by the image_loader function

        dpi : int, default None
            dpi for image animation

        total_layers : int, default None
            total number of layers to run (e.g. total_layers=3 will run layers 0, 1, 2)

        composite : boolean, default True
            Specifies whether or not to save composite images of all activations in a layer

        individuals : boolean, default True
            Specifies whether or not to save each activation as an individual image

        animation : boolean, default True
            Specifies whether or not to save an animation of activations

        composite_figsize : tuple, default (96, 96)
            Specifies matplotlib figsize for composite images

        animation_figsize : tuple, default (12, 12)
            Specifies matplotlib figsize for the activation animation

        resize : tuple, default (256,256)
            Specifies size to resize input image

        resize_animation : tuple, default (256, 256)
            Specifies size to resize images used in activation animation

        animation_interval : int, default 200
            Interval delay in miliseconds for animation

        folder_suffix : int, str default 1
            Suffix for the output directories created to save images

        return_acts : boolean, default True
            If True, function returns a list of activation tensors

    Returns:
        If return_acts=True, returns list of activation tensors
        Images are saved as .jpg files in the directories created
        Animations are saved as .mp4 files in the directory created

    """
    fn = PATH+filename

    if transformer is not None:
        i = transformer(open_image(fn))
        i = i[None]
        i = torch.from_numpy(i)
    else:
        i = image_loader(fn, resize=resize)

    i = i.cuda()
    model.eval()
    tmp_model = get_activation_layer(model)
    layer_outputs = tmp_model(V(i))

    layer_outputs = [i for i in layer_outputs if i.dim() == 4]
    if total_layers is not None:
        layer_outputs = layer_outputs[:total_layers]

    for i, layer in enumerate(layer_outputs):
        print('Processing Layer: ', i)
        features = layer.data
        images = features.cpu().numpy()[0]
        number_of_images = len(images)
        number_of_rows = np.sqrt(next_square(number_of_images)).astype('int')
        if composite:
            save_composite(images, PATH, i, rows=number_of_rows, cols=number_of_rows,
                        figsize=composite_figsize, suffix=folder_suffix)
        if individuals:
            save_individuals(images, PATH, i, suffix=folder_suffix)

    if animation:
        save_animation(layer_outputs, PATH, figsize=animation_figsize, dpi=dpi,
                resize=resize_animation, interval = animation_interval, suffix=folder_suffix)

    if return_acts:
        return layer_outputs


def image_loader(path, resize=(256,256)):
    """
    This function loads an image with PIL, resizes it, and converts to a torch tensor
    """
    img = PIL.Image.open(path)
    img = img.resize(resize)
    img = np.array(img, dtype=np.float32)
    img = np.einsum('ijk->kij', img)
    img = img[None]
    img = torch.from_numpy(img)
    return img


def next_square(n):
    if n%np.sqrt(n) == 0:
        return n
    else:
        return (np.int(np.sqrt(n))+1)**2


def return_sequential(layer_num, model):
    return nn.Sequential(
            *list(model.children())[:layer_num]
        )

class get_activation_layer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layer_models = []
        for i in range(len(self.model)):
             self.layer_models.append(return_sequential(i, self.model))
    def forward(self, x):
        self.outputs = []
        for i in range(len(self.model)):
            self.outputs.append(self.layer_models[i](x))
        return self.outputs


def save_animation(layer_outputs, PATH, figsize=(10,10), dpi=100, resize=(256,256),
                    interval=200, suffix=1):
    """
    This function takes a list of four dimensional activations and
    assembles them into an animation
    """
    dest = os.path.join(PATH, f'animation{suffix}/')
    os.makedirs(dest, exist_ok=True)

    if dpi:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = plt.figure(figsize=figsize)

    plt.axis('off')
    im_lst = []
    for i, layer in enumerate(layer_outputs):
        print('Animating Layer: ', i)
        features = layer.data
        images = features.cpu().numpy()[0]

        for j, im in enumerate(images):
            im_iter = PIL.Image.fromarray(im)
            im_iter = im_iter.resize(resize)
            im_iter = np.array(im_iter)
            im_plt = plt.imshow(im_iter, animated=True)
            im_lst.append([im_plt])
    ani = animation.ArtistAnimation(fig, im_lst, interval=interval)
    ani.save(f'{dest}cnn_animation.mp4')
    plt.close(fig)


def save_composite(ims, PATH, layer_num, rows=1, cols=None, figsize=(96,96), suffix=1):
    """
    This function takes a tensor of activations for a single layer and assembles
    them into a composite grid image
    """
    dest = os.path.join(PATH, f'composite{suffix}/')
    os.makedirs(dest, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = fig.add_subplot(rows, cols, i+1)
        plt.imshow(ims[i], interpolation=None, cmap=None)
        plt.axis('off')
        plt.subplots_adjust(hspace = 0.500)

    fig.savefig(f'{dest}layer{layer_num}.jpg', bbox_inches='tight')
    plt.close(fig)


def save_individuals(ims, PATH, layer_num, suffix=1):
    """
    This function takes a tensor of activations for a single layer and
    saves each slice as an individual image
    """
    dest = os.path.join(PATH, f'individuals{suffix}/')
    os.makedirs(dest, exist_ok=True)
    for j, im in enumerate(ims):
        fig = plt.figure()
        plt.imshow(im)
        plt.axis('off')
        fig.savefig(f'{dest}layer{layer_num}_image{j}.jpg', bbox_inches='tight')
        plt.close(fig)
