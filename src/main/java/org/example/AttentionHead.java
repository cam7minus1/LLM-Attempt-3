package org.example;

public class AttentionHead {

    Layer queryLayer;
    Layer keyLayer;
    Layer valueLayer;
    int embedSize;

    // Store last attention weights for backprop
    float[] lastAttentionWeights;
    float[][] lastQueries;
    float[][] lastKeys;
    float[][] lastValues;

    public AttentionHead(int embedSize) {
        this.embedSize = embedSize;
        // Each of Q, K, V is a layer that transforms embedSize -> embedSize
        queryLayer = new Layer(embedSize, embedSize, false);
        keyLayer = new Layer(embedSize, embedSize, false);
        valueLayer = new Layer(embedSize, embedSize, false);
    }

    public float[] forward(float[][] tokens) {
        int seqLen = tokens.length;
        float scale = (float) Math.sqrt(embedSize);

        lastQueries = new float[seqLen][];
        lastKeys = new float[seqLen][];
        lastValues = new float[seqLen][];

        // Compute Q, K, V for each token
        for (int i = 0; i < seqLen; i++) {
            lastQueries[i] = queryLayer.forwardPass(tokens[i]);
            lastKeys[i] = keyLayer.forwardPass(tokens[i]);
            lastValues[i] = valueLayer.forwardPass(tokens[i]);
        }

        // Use last token as query (we are predicting next token)
        float[] q = lastQueries[seqLen - 1];

        // Compute attention scores
        float[] scores = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            float dot = 0;
            for (int j = 0; j < embedSize; j++) {
                dot += q[j] * lastKeys[i][j];
            }
            scores[i] = dot / scale;
        }

        // Softmax scores into attention weights
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) if (s > max) max = s;
        float sum = 0;
        lastAttentionWeights = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            lastAttentionWeights[i] = (float) Math.exp(scores[i] - max);
            sum += lastAttentionWeights[i];
        }
        for (int i = 0; i < seqLen; i++) {
            lastAttentionWeights[i] /= sum;
        }

        // Weighted sum of value vectors
        float[] output = new float[embedSize];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < embedSize; j++) {
                output[j] += lastAttentionWeights[i] * lastValues[i][j];
            }
        }

        return output;
    }

    public float[] backward(float[] errorSignal, float learningRate) {
        int seqLen = lastAttentionWeights.length;

        // Gradient flows back through value weighted sum
        float[] valueGrad = new float[embedSize];
        for (int i = 0; i < seqLen; i++) {
            float[] vErr = new float[embedSize];
            for (int j = 0; j < embedSize; j++) {
                vErr[j] = errorSignal[j] * lastAttentionWeights[i];
            }
            valueLayer.backwardsPass(vErr, learningRate);
        }

        // Simplified gradient back through Q and K layers
        float[] qErr = new float[embedSize];
        for (int j = 0; j < embedSize; j++) {
            qErr[j] = errorSignal[j] * 0.01f;
        }
        float[] inputGrad = queryLayer.backwardsPass(qErr, learningRate);
        keyLayer.backwardsPass(qErr, learningRate);

        return inputGrad;
    }

    // Return all internal layers so Network can save/load them
    public Layer[] getLayers() {
        return new Layer[]{ queryLayer, keyLayer, valueLayer };
    }
}