<div align="center">
  
[1]: https://github.com/Akshat1661
[2]: https://www.linkedin.com/in/akshat-desai-10bba1235/


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]

</div>

# <div align="center">Image Caption Generator</div>

## Overview:
In the field of Machine Translation, we utilize two Recurrent Neural Networks (RNNs): the first serves as an encoder, while the second acts as a decoder. When translating text from one language to another, the encoder RNN processes the source text and compresses it into a single numerical vector. This vector is then fed into the decoder RNN, which generates the translated text in the target language. The intermediate vector, often referred to as a "thought vector," summarizes the source text, allowing the model to comprehend the entire input before proceeding with the translation. This design accommodates varying lengths of source and target texts.

In this project, we will replace the traditional encoder with an image recognition model. This image model will identify the contents of an image and produce a corresponding numerical vector, serving as the thought vector. This vector will then be used as input for the RNN decoder to generate descriptive text.

## Image-to-Text Generator Flow:

We will utilize the VGG16 model, which has been pre-trained for image classification tasks. Instead of employing the final classification layer, we will capture the output from the preceding layer. This results in a vector consisting of 4096 elements that encapsulates the contents of the image.

This vector will serve as the initial state for the Gated Recurrent Units (GRU). However, since the GRU has an internal state size of only 512, we need an intermediate fully connected (dense) layer to reduce the 4096-element vector to a 512-element vector.

The decoder will then use this initial state along with a start marker, "ssss," to initiate the generation of output words.

We will input this generated word into the decoder, which should ideally produce the word "brown," and the process will continue in this manner.

Ultimately, we will generate the text "big brown bear sitting eeee," where "eeee" signifies the end of the generated text.

## Dataset:
[Coco's Dataset](https://cocodataset.org/#home)<br>
COCO dataset contains many images with text-captions
<br>

## Implementation:

**Libraries:**  `NumPy` `pandas` `sklearn` `tensorflow` `seaborn` `keras` `Matplotlib`

## Pre-trained image model VGG16:
```
image_model = VGG16(include_top=True, weights='imagenet')
```
<br>

Above line creates an instance of the VGG16 model using the Keras API. This automatically downloads the required files if you don't have them already.

The VGG16 model has been pre-trained on the ImageNet dataset for the purpose of image classification. It consists of both a convolutional section and a fully connected (or dense) section used for classifying images.

When include_top=True, the entire VGG16 model is downloaded, which is approximately 528 MB in size. Conversely, setting include_top=False allows for the download of only the convolutional portion of the VGG16 model, resulting in a much smaller file size of around 57 MB.

Since we will utilize some of the fully connected layers from this pre-trained model, we will need to download the complete version. However, if you have a slow internet connection, you may consider modifying the code below to utilize the smaller pre-trained model without the classification layers.
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
Each image in the training set comes with at least five captions that describe its contents. The neural network will be trained using batches of transfer values from the images and sequences of integer tokens representing the captions. If we were to create matching NumPy arrays for the training set, we would either need to use only one caption per image, thereby disregarding valuable data, or we would have to duplicate the image transfer values for each caption, which would lead to excessive memory usage.

A more effective approach is to implement a custom data generator for Keras that produces batches of data with randomly selected transfer values and token sequences. This helper function provides a list of random token sequences corresponding to the images at specified indices in the training set.

The get_random_caption_tokens() generator function creates random batches of training data for the neural network training process. Additionally, the batch_generator() function generates random batches of training data by selecting data entirely at random for each batch, akin to sampling the training set with replacement. This means that the same data can be sampled multiple times within a single epoch, while some data might not be sampled at all. However, all data within a single batch will be unique.

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
The decoder produces output as a sequence of one-hot encoded arrays. To train the decoder, we need to provide the desired one-hot encoded arrays for its output and then utilize a loss function, such as cross-entropy, to guide the decoder in generating this expected output.

However, our dataset consists of integer tokens rather than one-hot encoded arrays. Since each one-hot encoded array contains 10,000 elements, converting the entire dataset to this format would be highly inefficient. We could perform this conversion from integers to one-hot arrays within the batch_generator() function mentioned earlier.

A more efficient approach is to use a sparse cross-entropy loss function, which automatically handles the conversion from integers to one-hot encoded arrays internally.

While we have often employed the Adam optimizer in previous tutorials, it has shown signs of divergence in some of our experiments with Recurrent Neural Networks. In these cases, the RMSprop optimizer tends to yield better results.


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


![imagecaption](https://github.com/user-attachments/assets/23774382-96da-4ac4-b33e-0f5c69e0975c)


### Learnings:
`Convolutional Neural Networks`
`Transfer Learnign`
`Fine-tuning`
`Neural Networks`
`dash-plotly`
`VGG16`

### Feedback

If you have any feedback, please reach out at akshat.desai.754@gmail.com

[1]: https://github.com/Akshat1661
[2]: https://www.linkedin.com/in/akshat-desai-10bba1235/


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]

# Image_Caption_Generator
