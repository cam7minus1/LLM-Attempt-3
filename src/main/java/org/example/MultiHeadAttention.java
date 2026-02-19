package org.example;

public class MultiHeadAttention {

    AttentionHead[] heads;
    Layer outputProjection;
    int embedSize;
    int numHeads;

    public MultiHeadAttention(int embedSize, int numHeads) {
        this.embedSize = embedSize;
        this.numHeads = numHeads;
        this.heads = new AttentionHead[numHeads];

        for (int i = 0; i < numHeads; i++) {
            heads[i] = new AttentionHead(embedSize);
        }

        // Projects concatenated head outputs back to embedSize
        outputProjection = new Layer(embedSize, embedSize * numHeads, false);
    }

    public float[] forward(float[][] tokens) {
        float[] concatenated = new float[embedSize * numHeads];

        for (int h = 0; h < numHeads; h++) {
            float[] headOutput = heads[h].forward(tokens);
            System.arraycopy(headOutput, 0, concatenated, h * embedSize, embedSize);
        }

        return outputProjection.forwardPass(concatenated);
    }

    public float[] backward(float[] errorSignal, float learningRate) {
        float[] projGrad = outputProjection.backwardsPass(errorSignal, learningRate);

        float[] inputGrad = new float[embedSize];
        for (int h = 0; h < numHeads; h++) {
            float[] headErr = new float[embedSize];
            System.arraycopy(projGrad, h * embedSize, headErr, 0, embedSize);
            float[] hGrad = heads[h].backward(headErr, learningRate);
            for (int i = 0; i < embedSize; i++) {
                inputGrad[i] += hGrad[i];
            }
        }

        return inputGrad;
    }

    public Layer[] getLayers() {
        // Collect all layers from all heads plus output projection
        Layer[] allLayers = new Layer[numHeads * 3 + 1];
        int idx = 0;
        for (AttentionHead head : heads) {
            for (Layer l : head.getLayers()) {
                allLayers[idx++] = l;
            }
        }
        allLayers[idx] = outputProjection;
        return allLayers;
    }
}