package org.example;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;

public class Network {

    Layer[] layers;

    public Network(int inputSize, int hiddenSize, int outputSize, int numHiddenLayers) {

        layers = new Layer[numHiddenLayers + 1];

        // First hidden layer
        layers[0] = new Layer(hiddenSize, inputSize, false);

        // Middle hidden layers
        for (int i = 1; i < numHiddenLayers; i++) {
            layers[i] = new Layer(hiddenSize, hiddenSize, false);
        }

        // Output layer
        layers[numHiddenLayers] = new Layer(outputSize, hiddenSize, true);
    }

    public float[] softMaxActivation(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : input) {
            if (v > max) max = v;
        }

        double sum = 0.0;
        float[] result = new float[input.length];

        for (int i = 0; i < input.length; i++) {
            sum += Math.exp(input[i] - max);
        }

        for (int i = 0; i < input.length; i++) {
            result[i] = (float)(Math.exp(input[i] - max) / sum);
        }

        return result;
    }

    public float[] forward(float[] input) {
        float[] previous = input;

        for (Layer layer : layers) {
            previous = layer.forwardPass(previous);
        }

        return softMaxActivation(previous);
    }

    public void backwards(float[] blames, float learningRate) {
        float[] prevErrorSignal = blames;

        for (int i = layers.length - 1; i >= 0; i--) {
            prevErrorSignal = layers[i].backwardsPass(prevErrorSignal, learningRate);
        }
    }

    // ---------- SAVE / LOAD ----------

    static class SavedNetwork {
        int inputSize;
        int hiddenSize;
        int outputSize;
        int numHiddenLayers;
        float[][][] weights; // [layer][neuron][weight]
        float[][] biases;    // [layer][neuron]
    }

    public void saveWeights(String path) {
        try {
            SavedNetwork sn = new SavedNetwork();
            sn.numHiddenLayers = layers.length - 1;
            sn.inputSize = layers[0].getInputSize();
            sn.hiddenSize = layers[0].getNumNeurons();
            sn.outputSize = layers[layers.length - 1].getNumNeurons();

            sn.weights = new float[layers.length][][];
            sn.biases = new float[layers.length][];

            for (int l = 0; l < layers.length; l++) {
                Layer layer = layers[l];
                int n = layer.getNumNeurons();
                sn.weights[l] = new float[n][];
                sn.biases[l] = new float[n];

                for (int i = 0; i < n; i++) {
                    Neuron neuron = layer.getNeurons()[i];
                    float[] w = neuron.getWeights();
                    sn.weights[l][i] = new float[w.length];
                    System.arraycopy(w, 0, sn.weights[l][i], 0, w.length);
                    sn.biases[l][i] = neuron.getBias();
                }
            }

            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter fw = new FileWriter(path)) {
                gson.toJson(sn, fw);
            }

            System.out.println("Saved weights to: " + path);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Network loadOrCreate(String path, int inputSize, int hiddenSize, int outputSize, int numHiddenLayers) {
        File f = new File(path);
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        if (!f.exists()) {
            System.out.println("No weights file found. Creating new network.");
            return new Network(inputSize, hiddenSize, outputSize, numHiddenLayers);
        }

        try (FileReader fr = new FileReader(f)) {
            SavedNetwork sn = gson.fromJson(fr, SavedNetwork.class);

            Network net = new Network(inputSize, hiddenSize, outputSize, numHiddenLayers);

            // Copy all layers except possibly expanded output
            int minLayers = Math.min(net.layers.length, sn.weights.length);

            for (int l = 0; l < minLayers; l++) {
                Layer layer = net.layers[l];
                int numNeuronsCurrent = layer.getNumNeurons();
                int numNeuronsSaved = sn.weights[l].length;
                int neuronsToCopy = Math.min(numNeuronsCurrent, numNeuronsSaved);

                for (int i = 0; i < neuronsToCopy; i++) {
                    Neuron neuron = layer.getNeurons()[i];

                    float[] savedW = sn.weights[l][i];
                    float[] w = neuron.getWeights();
                    int wToCopy = Math.min(w.length, savedW.length);
                    System.arraycopy(savedW, 0, w, 0, wToCopy);
                    neuron.setBias(sn.biases[l][i]);
                }
                // Extra neurons (if output expanded) keep random init
            }

            System.out.println("Loaded weights from: " + path);
            return net;

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to load weights. Creating new network.");
            return new Network(inputSize, hiddenSize, outputSize, numHiddenLayers);
        }
    }
}