package org.example;

public class TransformerBlock {

    MultiHeadAttention attention;
    Layer feedForward1;
    Layer feedForward2;
    int embedSize;

    public TransformerBlock(int embedSize, int numHeads, int ffSize) {
        this.embedSize = embedSize;
        this.attention = new MultiHeadAttention(embedSize, numHeads);
        this.feedForward1 = new Layer(ffSize, embedSize, false);
        this.feedForward2 = new Layer(embedSize, ffSize, false);
    }

    public float[] forward(float[][] tokens) {
        // 1. Attention
        float[] attnOutput = attention.forward(tokens);

        // 2. Residual connection - add last token embedding to attention output
        float[] lastToken = tokens[tokens.length - 1];
        float[] residual1 = new float[embedSize];
        for (int i = 0; i < embedSize; i++) {
            residual1[i] = attnOutput[i] + lastToken[i];
        }

        // 3. Feed forward
        float[] ff1 = feedForward1.forwardPass(residual1);
        float[] ff2 = feedForward2.forwardPass(ff1);

        // 4. Second residual connection
        float[] output = new float[embedSize];
        for (int i = 0; i < embedSize; i++) {
            output[i] = ff2[i] + residual1[i];
        }

        return output;
    }

    public float[] backward(float[] errorSignal, float learningRate) {
        // Backprop through second residual
        float[] ff2Err = errorSignal.clone();

        // Backprop through ff2
        float[] ff1Err = feedForward2.backwardsPass(ff2Err, learningRate);

        // Backprop through ff1
        float[] residualErr = feedForward1.backwardsPass(ff1Err, learningRate);

        // Add residual gradient
        for (int i = 0; i < embedSize; i++) {
            residualErr[i] += errorSignal[i];
        }

        // Backprop through attention
        return attention.backward(residualErr, learningRate);
    }

    public Layer[] getLayers() {
        Layer[] attnLayers = attention.getLayers();
        Layer[] allLayers = new Layer[attnLayers.length + 2];
        System.arraycopy(attnLayers, 0, allLayers, 0, attnLayers.length);
        allLayers[attnLayers.length] = feedForward1;
        allLayers[attnLayers.length + 1] = feedForward2;
        return allLayers;
    }
}