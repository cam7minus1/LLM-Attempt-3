package org.example;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Network {

    TransformerBlock[] transformerBlocks;
    Layer outputLayer;
    int embedSize;
    int vocabSize;

    public Network(int embedSize, int vocabSize, int numHeads, int numBlocks, int ffSize) {
        this.embedSize = embedSize;
        this.vocabSize = vocabSize;

        transformerBlocks = new TransformerBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++) {
            transformerBlocks[i] = new TransformerBlock(embedSize, numHeads, ffSize);
        }

        outputLayer = new Layer(vocabSize, embedSize, true);
    }

    public float[] forward(float[][] tokens) {
        float[][] current = tokens;

        for (TransformerBlock block : transformerBlocks) {
            float[] blockOutput = block.forward(current);

            float[][] next = new float[current.length][];
            for (int i = 0; i < current.length - 1; i++) {
                next[i] = current[i];
            }
            next[current.length - 1] = blockOutput;
            current = next;
        }

        float[] logits = outputLayer.forwardPass(current[current.length - 1]);
        return softmax(logits);
    }

    public void backward(float[] errorSignal, float learningRate) {
        float[] err = outputLayer.backwardsPass(errorSignal, learningRate);

        for (int i = transformerBlocks.length - 1; i >= 0; i--) {
            err = transformerBlocks[i].backward(err, learningRate);
        }
    }

    public float[] softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : input) if (v > max) max = v;

        double sum = 0;
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = (float) Math.exp(input[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < input.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    // ---------- COLLECT ALL LAYERS ----------

    private Layer[] getAllTransformerLayers() {
        List<Layer> all = new ArrayList<>();
        for (TransformerBlock block : transformerBlocks) {
            for (Layer l : block.getLayers()) {
                all.add(l);
            }
        }
        return all.toArray(new Layer[0]);
    }

    // ---------- SAVE ----------

    static class SavedNetwork {
        int embedSize;
        int vocabSize;
        int numHeads;
        int numBlocks;
        int ffSize;
        float[][][] transformerWeights;
        float[][] transformerBiases;
        float[][] outputWeights;
        float[] outputBiases;
    }

    public void saveWeights(String path) {
        try {
            SavedNetwork sn = new SavedNetwork();
            sn.embedSize = embedSize;
            sn.vocabSize = vocabSize;

            // Save transformer layers separately from output layer
            Layer[] tLayers = getAllTransformerLayers();
            sn.transformerWeights = new float[tLayers.length][][];
            sn.transformerBiases = new float[tLayers.length][];

            for (int l = 0; l < tLayers.length; l++) {
                Layer layer = tLayers[l];
                int n = layer.getNumNeurons();
                sn.transformerWeights[l] = new float[n][];
                sn.transformerBiases[l] = new float[n];

                for (int i = 0; i < n; i++) {
                    Neuron neuron = layer.getNeurons()[i];
                    float[] w = neuron.getWeights();
                    sn.transformerWeights[l][i] = new float[w.length];
                    System.arraycopy(w, 0, sn.transformerWeights[l][i], 0, w.length);
                    sn.transformerBiases[l][i] = neuron.getBias();
                }
            }

            // Save output layer separately so it can expand independently
            int outNeurons = outputLayer.getNumNeurons();
            sn.outputWeights = new float[outNeurons][];
            sn.outputBiases = new float[outNeurons];

            for (int i = 0; i < outNeurons; i++) {
                Neuron neuron = outputLayer.getNeurons()[i];
                float[] w = neuron.getWeights();
                sn.outputWeights[i] = new float[w.length];
                System.arraycopy(w, 0, sn.outputWeights[i], 0, w.length);
                sn.outputBiases[i] = neuron.getBias();
            }

            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter fw = new FileWriter(path)) {
                gson.toJson(sn, fw);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ---------- LOAD ----------

    public static Network loadOrCreate(String path, int embedSize, int vocabSize, int numHeads, int numBlocks, int ffSize) {
        File f = new File(path);
        if (!f.exists()) {
            System.out.println("No weights file found. Creating new network.");
            return new Network(embedSize, vocabSize, numHeads, numBlocks, ffSize);
        }

        try (FileReader fr = new FileReader(f)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            SavedNetwork sn = gson.fromJson(fr, SavedNetwork.class);

            Network net = new Network(embedSize, vocabSize, numHeads, numBlocks, ffSize);

            // Load transformer layers
            Layer[] tLayers = net.getAllTransformerLayers();
            int minLayers = Math.min(tLayers.length, sn.transformerWeights.length);

            for (int l = 0; l < minLayers; l++) {
                Layer layer = tLayers[l];
                int neuronsToCopy = Math.min(layer.getNumNeurons(), sn.transformerWeights[l].length);

                for (int i = 0; i < neuronsToCopy; i++) {
                    Neuron neuron = layer.getNeurons()[i];
                    float[] savedW = sn.transformerWeights[l][i];
                    float[] w = neuron.getWeights();
                    int wToCopy = Math.min(w.length, savedW.length);
                    System.arraycopy(savedW, 0, w, 0, wToCopy);
                    neuron.setBias(sn.transformerBiases[l][i]);
                }
            }

            // Load output layer with expansion support
            // Only copy neurons that existed before, new vocab words keep random init
            int savedOutputSize = sn.outputWeights.length;
            int currentOutputSize = net.outputLayer.getNumNeurons();
            int neuronsToCopy = Math.min(currentOutputSize, savedOutputSize);

            if (savedOutputSize < currentOutputSize) {
                System.out.println("Vocab expanded from " + savedOutputSize + " to " + currentOutputSize
                        + " — new words will use random init and improve with training.");
            }

            for (int i = 0; i < neuronsToCopy; i++) {
                Neuron neuron = net.outputLayer.getNeurons()[i];
                float[] savedW = sn.outputWeights[i];
                float[] w = neuron.getWeights();
                int wToCopy = Math.min(w.length, savedW.length);
                System.arraycopy(savedW, 0, w, 0, wToCopy);
                neuron.setBias(sn.outputBiases[i]);
            }

            System.out.println("Loaded weights from: " + path);
            return net;

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to load weights. Creating new network.");
            return new Network(embedSize, vocabSize, numHeads, numBlocks, ffSize);
        }
    }
}