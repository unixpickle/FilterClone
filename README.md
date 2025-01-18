# FilterClone

I trained neural networks to apply intelligent "filters" to images. These neural networks are trained by cloning popular Snapchat lenses. Here are some examples:

| Input        | Aged Filter   | Cartoon Filter   |
|--------------|---------------|------------------|
| ![Input](examples/1_in.jpg) | ![Aged](examples/1_old.jpg) | ![Cartoon](examples/1_cartoon.jpg) |
| ![Input](examples/2_in.jpg) | ![Aged](examples/2_old.jpg) | ![Cartoon](examples/2_cartoon.jpg) |
| ![Input](examples/3_in.jpg) | ![Aged](examples/3_old.jpg) | ![Cartoon](examples/3_cartoon.jpg) |
| ![Input](examples/4_in.jpg) | ![Aged](examples/4_old.jpg) | ![Cartoon](examples/4_cartoon.jpg) |

# How it works

## Dataset

I created a dataset of around 100K input-output pairs, where the input is an image and the output is the result of applying a filter. For input images, I reused my [filtering code](https://github.com/unixpickle/laion-icons) for the LAION dataset to scrape images which are likely to contain human faces. To get output images, I reverse engineered pieces of Snapchat's Chrome extension to apply Snapchat "lenses" to the input images.

## Model architecture

Once I had a dataset, I trained convolutional UNet models on the input-output pairs. I train the model with a regression object&mdash;for a given input, it is trying to predict the mean output. Note that this is different than a diffusion model, where we are actually _sampling from a distribution_ of possible outputs.

Regression can be problematic when the Snapchat lens is nondeterministic. For example, let's say a lens might either choose to make your beard white or black, and this choice is random. With a regression model, the best we can do is always predict the _average_ of these colors, i.e. gray. Even if the lens itself is deterministic, we might introduce nondeterminism when applying data augmentation (such as random crops) to the dataset.

To allow the model to be _slightly_ generative, and therefore deal with some nondeterminism, I introduce a single VQ latent code. This makes our model a very weak [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder), allowing us to sample a handful of possible outputs for a given input. This is a _tiny_ amount of generative power, but it's still cool to see:

![Sample grid of different old photos of a celebrity](examples/grid.jpg)

# Pretrained models

Here are pre-trained models for some filters:

 * [Aged/old filter](https://data.aqnichol.com/FilterClone/old_model.plist)
 * [Cartoon filter](https://data.aqnichol.com/FilterClone/cartoon_model.plist)

# Applying filters

You can run a web UI to apply a filter like so:

```
$ swift run -c release FilterApply --model-path <my_model_file.plist>
```

By default, this will listen on `http://localhost:1235`.
