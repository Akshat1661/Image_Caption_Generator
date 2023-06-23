<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Image Caption Generator</div>
<div align="center"><img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/image_captioning.gif?raw=true"></div>

## Overview:
In Machine Translation, we use two Recurrent Neural Networks (RNN), the first called an encoder and the second called a decoder.
If we want to translate text from one human language to another, The first RNN encodes the source-text as a single vector of numbers and the second RNN decodes this vector into the destination-text. The intermediate vector between the encoder and decoder is a kind of summary of the source-text, which is sometimes called a "thought-vector". The reason for using this intermediate summary-vector is to understand the whole source-text before it is being translated. This also allows for the source- and destination-texts to have different lengths.
In this project, we will replace the encoder with an image-recognition model.
The image-model recognizes what the image contains and outputs that as a vector of numbers the "thought-vector" or summary-vector, which is then input to a Recurrent Neural Network that decodes this vector into text.

## Flowchart:
<br>
<img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/intro.PNG?raw=true">
<br>

- We will use the VGG16 model that has been pre-trained for classifying images. But instead of using the last classification layer, we will redirect the output of the previous layer. This gives us a vector with 4096 elements that summarizes the image-contents 
- We will use this vector as the initial state of the Gated Recurrent Units (GRU). However, the internal state-size of the GRU is only 512, so we need an intermediate fully-connected (dense) layer to map the vector with 4096 elements down to a vector with only 512 elements.
- The decoder then uses this initial-state together with a start-marker "ssss" to begin producing output words.
- Then we input this word into the decoder and hopefully we get the word "brown" out, and so on.
- Finally we have generated the text "big brown bear sitting eeee" where "eeee" marks the end of the text.

## Dataset:
[Coco's Dataset](https://cocodataset.org/#home)<br>
COCO dataset contains many images with text-captions
<br>
#### Example Image:
<img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/example.PNG?raw=true">



## Implementation:

**Libraries:**  `NumPy` `pandas` `sklearn` `tensorflow` `seaborn` `keras` `Matplotlib`

## Pre-trained image model VGG16:
```
image_model = VGG16(include_top=True, weights='imagenet')
```
<br>

Above line creates an instance of the VGG16 model using the Keras API. This automatically downloads the required files if you don't have them already.

The VGG16 model was pre-trained on the ImageNet data-set for classifying images. The VGG16 model contains a convolutional part and a fully-connected (or dense) part which is used for the image classification.

If include_top=True then the whole VGG16 model is downloaded which is about 528 MB. If include_top=False then only the convolutional part of the VGG16 model is downloaded which is just 57 MB.

We will use some of the fully-connected layers in this pre-trained model, so we have to download the full model, but if you have a slow internet connection, then you can try and modify the code below to use the smaller pre-trained model without the classification layers.
<br>

#### Model summary:
```
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
```
<br>

**We will use the output of the layer prior to the final classification-layer which is named `fc2`. This is a fully-connected (or dense) layer.**
<br>

```
transfer_layer = image_model.get_layer('fc2')
```

We call it the "transfer-layer" because we will transfer its output to another model that creates the image captions.
<br>

```
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
```




## Processing all the images:
We now make functions for processing all images in the data-set using the pre-trained image-model and saving the transfer-values in a cache-file so they can be reloaded quickly.

We effectively create a new data-set of the transfer-values. This is because it takes a long time to process an image in the VGG16 model. We will not be changing all the parameters of the VGG16 model, so every time it processes an image, it gives the exact same result. We need the transfer-values to train the image-captioning model for many epochs, so we save a lot of time by calculating the transfer-values once and saving them in a cache-file.
<br>
`process_images()` , `process_images_train()` , and `process_images_val()` are the helper functions used for processing the images.<br>
Checkout the [Notebook](https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/Scripts/Image_caption_generator.ipynb) for more details.



## Tokenizer:
Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network. The first step is to convert text-words into so-called integer-tokens. The second step is to convert integer-tokens into vectors of floating-point numbers using a so-called embedding-layer. 
Before we can start processing the text, we first need to mark the beginning and end of each text-sequence with unique words that most likely aren't present in the data.
`mark_captions` is the helper-function wraps all text-strings with start and end markers.
<br>
```
['ssss Closeup of bins of food that include broccoli and bread. eeee',
 'ssss A meal is presented in brightly colored plastic trays. eeee',
 'ssss there are containers filled with different kinds of foods eeee',
 'ssss Colorful dishes holding meat, vegetables, fruit, and bread. eeee',
 'ssss A bunch of trays that have different food. eeee']
```
This is how the captions look without the start- and end-markers.
<br>
`TokenizerWrap()` , `token_to_word()` , `tokens_to_string()` , `captions_to_tokens()` are the fuctions used in the process of tokenization.

### Steps involed in tokenization:
- Wrap the Tokenizer-class from Keras with more functionality:
```
tokenizer = TokenizerWrap(texts=captions_train_flat,num_words=num_words)
```
- Get the integer-token for the start-marker (the word "ssss"):
```
token_start = tokenizer.word_index[mark_start.strip()]
```
- Get the integer-token for the end-marker (the word "eeee"):
```
token_end = tokenizer.word_index[mark_end.strip()]
```
- Convert all the captions from the training-set to sequences of integer-tokens:
```
tokens_train = tokenizer.captions_to_tokens(captions_train_marked)
```



## Data Generator:
Each image in the training-set has at least 5 captions describing the contents of the image. The neural network will be trained with batches of transfer-values for the images and sequences of integer-tokens for the captions. If we were to have matching numpy arrays for the training-set, we would either have to only use a single caption for each image and ignore the rest of this valuable data, or we would have to repeat the image transfer-values for each of the captions, which would waste a lot of memory.

A better solution is to create a custom data-generator for Keras that will create a batch of data with randomly selected transfer-values and token-sequences.
This helper-function returns a list of random token-sequences for the images with the given indices in the training-set.
<br>
`get_random_caption_tokens()` generator function creates random batches of training-data for use in training the neural network.
<br>
`batch_generator()`  Generator function for creating random batches of training-data. It selects the data completely randomly for each batch, corresponding to sampling of the training-set with replacement. This means it is possible to sample the same data multiple times within a single epoch - and it is also possible that some data is not sampled at all within an epoch. However, all the data should be unique within a single batch.

#### Steps:
- Create an instance of the data-generator:
```
generator = batch_generator(batch_size=batch_size)
```
- Test the data-generator by creating a batch of data:
```
batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]
```
- Example of the transfer-values for the first image in the batch:
```
batch_x['transfer_values_input'][0]
```
```
array([0.   , 0.   , 1.483, ..., 0.   , 0.   , 0.813], dtype=float16)
```
- Example of the token-sequence for the first image in the batch. This is the input to the decoder-part of the neural network:
```
batch_x['decoder_input'][0]
```
```
array([   2,    1,   21,   80,   13,   34,  315,    1,   69,   20,   12,
          1, 1083,    3,    0,    0,    0,    0,    0,    0,    0],
      dtype=int32)
```
- The token-sequence for the output of the decoder:
```
batch_y['decoder_output'][0]
```
```
array([   1,   21,   80,   13,   34,  315,    1,   69,   20,   12,    1,
       1083,    3,    0,    0,    0,    0,    0,    0,    0,    0],
      dtype=int32)
```



## Steps per Epoch:
One epoch is a complete processing of the training-set. We would like to process each image and caption pair only once per epoch. However, because each batch is chosen completely at random in the above batch-generator, it is possible that an image occurs in multiple batches within a single epoch, and it is possible that some images may not occur in any batch at all within a single epoch.

Nevertheless, we still use the concept of an 'epoch' to measure approximately how many iterations of the training-data we have processed. But the data-generator will generate batches for eternity, so we need to manually calculate the approximate number of batches required per epoch.

## Create the Recurrent Neural Network:
We will now create the Recurrent Neural Network (RNN) that will be trained to map the vectors with transfer-values from the image-recognition model into sequences of integer-tokens that can be converted into text.
We are using the functional model from Keras to build this neural network, because it allows more flexibility.
<br>

We want to use the transfer-values to initialize the internal states of the GRU units. This informs the GRU units of the contents of the images. The transfer-values are vectors of length 4096 but the size of the internal states of the GRU units are only 512, so we use a fully-connected layer to map the vectors from 4096 to 512 elements.
```
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')
decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')
```
We use a `tanh` activation function to limit the output of the mapping between -1 and 1, otherwise this does not seem to work.
```
decoder_input = Input(shape=(None, ), name='decoder_input')
```
This is the input for token-sequences to the decoder. Using None in the shape means that the token-sequences can have arbitrary lengths.
```
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')
```
This is the embedding-layer which converts sequences of integer-tokens to sequences of vectors.
```
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
```
This creates the 3 GRU layers of the decoder. Note that they all return sequences because we ultimately want to output a sequence of integer-tokens that can be converted into a text-sequence.
<br>
Each "word" is encoded as a vector of length state_size. We need to convert this into sequences of integer-tokens that can be interpreted as words from our vocabulary.
One way of doing this is to convert the GRU output to a one-hot encoded array. It works but it is extremely wasteful, because for a vocabulary of e.g. 10000 words we need a vector with 10000 elements, so we can select the index of the highest element to be the integer-token.
```
decoder_dense = Dense(num_words,
                      activation='softmax',
                      name='decoder_output')
```

## Connect and Create the Training Model:
The decoder is built using the functional API of Keras, which allows more flexibility in connecting the layers e.g. to have multiple inputs. This is useful e.g. if you want to connect the image-model directly with the decoder instead of using pre-calculated transfer-values.

`connect_decoder()` function connects all the layers of the decoder to some input of transfer-values.

```
decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])
```


## Compile the model:
The output of the decoder is a sequence of one-hot encoded arrays. In order to train the decoder we need to supply the one-hot encoded arrays that we desire to see on the decoder's output, and then use a loss-function like cross-entropy to train the decoder to produce this desired output.

However, our data-set contains integer-tokens instead of one-hot encoded arrays. Each one-hot encoded array has 10000 elements so it would be extremely wasteful to convert the entire data-set to one-hot encoded arrays. We could do this conversion from integers to one-hot arrays in the batch_generator() above.

A better way is to use a so-called sparse cross-entropy loss-function, which does the conversion internally from integers to one-hot encoded arrays.

We have used the Adam optimizer in many of the previous tutorials, but it seems to diverge in some of these experiments with Recurrent Neural Networks. RMSprop seems to work much better for these.
```
decoder_model.compile(optimizer=RMSprop(lr=1e-3),
                      loss='sparse_categorical_crossentropy')
```


## Train the model:
Now we will train the decoder so it can map transfer-values from the image-model to sequences of integer-tokens for the captions of the images.
```
decoder_model.fit(x=generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=20,
                  callbacks=callbacks)
```
## Generate captions:
`generate_caption` function loads an image and generates a caption using the model we have trained.

### Results:
<img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/predict1.PNG?raw=true" width="40%">  
<img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/predict3.PNG?raw=true"><img src="https://github.com/Pradnya1208/Image-Caption-Generator/blob/main/output/predict4.PNG?raw=true">


### Learnings:
`Convolutional Neural Networks`
`Transfer Learnign`
`Fine-tuning`
`Neural Networks`
`dash-plotly`
`VGG16`






## References:
[Hvass Laboratories](https://www.hvass-labs.org/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner








[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]





# Image_Caption_Generator
