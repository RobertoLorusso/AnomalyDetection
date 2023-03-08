Data
The dataset contains 5,000 Time Series examples (obtained with ECG) with 140 timesteps. Each sequence corresponds to a single heartbeat from a single patient with congestive heart failure.

An electrocardiogram (ECG or EKG) is a test that checks how your heart is functioning by measuring the electrical activity of the heart. With each heart beat, an electrical impulse (or wave) travels through your heart. This wave causes the muscle to squeeze and pump blood from the heart. Source

We have 5 types of hearbeats (classes):

1. Normal (N)
2. Premature Ventricular Contraction (PVC)
3. R-on-T Premature Ventricular Contraction (R-on-T PVC)
4. Supra-ventricular Premature or Ectopic Beat (SP or EB)
5. Unclassified Beat (UB).

Assuming a healthy heart and a typical rate of 70 to 75 beats per minute, each cardiac cycle, or heartbeat, takes about 0.8 seconds to complete the cycle. Frequency: 60–100 per minute (Humans) Duration: 0.6–1 second (Humans)

An LSTM Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture.

For a given dataset of sequences, an encoder-decoder LSTM is configured to read the input sequence, encode it, decode it, and recreate it. The performance of the model is evaluated based on the model’s ability to recreate the input sequence.

Once the model achieves a desired level of performance recreating the sequence, the decoder part of the model may be removed, leaving just the encoder model. This model can then be used to encode input sequences to a fixed-length vector.

The resulting vectors can then be used in a variety of applications, not least as a compressed representation of the sequence as an input to another supervised learning model.

The Autoencoder’s job is to get some input data, pass it through the model, and obtain a reconstruction of the input. The reconstruction should match the input as much as possible. The trick is to use a small number of parameters, so your model learns a compressed representation of the data.

In a sense, Autoencoders try to learn only the most important features (compressed version) of the data. Here, we’ll have a look at how to feed Time Series data to an Autoencoder. We’ll use a couple of LSTM layers (hence the LSTM Autoencoder) to capture the temporal dependencies of the data.

To classify a sequence as normal or an anomaly, we’ll pick a threshold above which a heartbeat is considered abnormal.

Reconstruction Loss
When training an Autoencoder, the objective is to reconstruct the input as best as possible. This is done by minimizing a loss function (just like in supervised learning). This function is known as reconstruction loss. Cross-entropy loss and Mean squared error are common examples.

Anomaly Detection in ECG Data
We'll use normal heartbeats as training data for our model and record the reconstruction loss. But first, we need to prepare the data:

Data Preprocessing
Let's get all normal heartbeats and drop the target (class) column:

We optimize the parameters of our Autoencoder model in such way that a special kind of error - reconstruction error is minimized. In practice, the traditional squared error is often used:

![image.png](data:image/png;base64,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)
