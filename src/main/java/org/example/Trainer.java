package org.example;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.util.*;

public class Trainer {

    Network network;
    CrossCatagoricalEntropyLoss ccel;
    float learningRate;
    int outputSize;
    int batchSize;

    Map<String, float[]> embeddings;
    List<String> vocabList;
    Map<String, Integer> wordToIndex;

    public Trainer(Network network, float learningRate, int outputSize, int batchSize) {
        this.network = network;
        this.ccel = new CrossCatagoricalEntropyLoss();
        this.learningRate = learningRate;
        this.outputSize = outputSize;
        this.batchSize = batchSize;
    }

    public void trainRealData(String cleanedFilePath, String embeddingsPath, int epochs, Config config) {
        try {
            // 1. Load embeddings
            Gson gson = new Gson();
            Map<String, Object> jsonData = gson.fromJson(
                    new FileReader(embeddingsPath),
                    new TypeToken<Map<String, Object>>(){}.getType()
            );

            Map<String, List<Double>> loaded = (Map<String, List<Double>>) jsonData.get("embeddings");

            embeddings = new HashMap<>();
            vocabList = new ArrayList<>();
            wordToIndex = new HashMap<>();

            List<String> sortedWords = new ArrayList<>(loaded.keySet());
            Collections.sort(sortedWords);

            for (String word : sortedWords) {
                List<Double> vecD = loaded.get(word);
                float[] vec = new float[config.embedSize];
                for (int j = 0; j < config.embedSize; j++) {
                    vec[j] = vecD.get(j).floatValue();
                }
                embeddings.put(word, vec);
                vocabList.add(word);
            }

            for (int i = 0; i < vocabList.size(); i++) {
                wordToIndex.put(vocabList.get(i), i);
            }

            System.out.println("Embeddings loaded. Starting training...");

            // 2. Train epoch by epoch, reading file in batches
            for (int e = 0; e < epochs; e++) {
                float epochLoss = 0f;
                int totalExamples = 0;

                try (BufferedReader br = new BufferedReader(new FileReader(cleanedFilePath))) {

                    List<String> wordBuffer = new ArrayList<>();
                    String line;

                    while (true) {
                        line = br.readLine();

                        if (line != null) {
                            String[] split = line.toLowerCase().split("[^a-zA-Z0-9']+");
                            for (String w : split) {
                                if (!w.isEmpty() && embeddings.containsKey(w)) {
                                    wordBuffer.add(w);
                                }
                            }
                            if (embeddings.containsKey("<END>")) {
                                wordBuffer.add("<END>");
                            }
                        }

                        boolean fileEnded = (line == null);

                        while (wordBuffer.size() >= config.windowSize + batchSize ||
                                (fileEnded && wordBuffer.size() > config.windowSize)) {

                            List<float[][]> batchInputs = new ArrayList<>();
                            List<float[]> batchLabels = new ArrayList<>();

                            int limit = Math.min(batchSize, wordBuffer.size() - config.windowSize);

                            for (int i = 0; i < limit; i++) {
                                float[][] tokenWindow = new float[config.windowSize][];
                                for (int p = 0; p < config.windowSize; p++) {
                                    tokenWindow[p] = embeddings.get(wordBuffer.get(i + p));
                                }

                                String nextWord = wordBuffer.get(i + config.windowSize);
                                Integer idx = wordToIndex.get(nextWord);
                                if (idx != null && idx < outputSize) {
                                    float[] label = new float[outputSize];
                                    label[idx] = 1.0f;
                                    batchInputs.add(tokenWindow);
                                    batchLabels.add(label);
                                }
                            }

                            for (int i = 0; i < batchInputs.size(); i++) {
                                float[] output = network.forward(batchInputs.get(i));
                                float loss = ccel.calculate(output, batchLabels.get(i));
                                epochLoss += loss;
                                totalExamples++;

                                float[] errorSignal = new float[output.length];
                                for (int j = 0; j < output.length; j++) {
                                    errorSignal[j] = output[j] - batchLabels.get(i)[j];
                                }
                                network.backward(errorSignal, learningRate);
                            }

                            wordBuffer.subList(0, limit).clear();
                        }

                        if (fileEnded) break;
                    }
                }

                float avgLoss = totalExamples > 0 ? epochLoss / totalExamples : 0;
                System.out.println("Epoch " + e + " - Avg Loss: " + avgLoss + " - Examples: " + totalExamples);
                network.saveWeights("src/main/resources/networkWeights.json");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}