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

    Map<String, float[]> embeddings;
    List<String> vocabList;
    Map<String, Integer> wordToIndex;

    public Trainer(Network network, float learningRate, int outputSize){
        this.network = network;
        this.ccel = new CrossCatagoricalEntropyLoss();
        this.learningRate = learningRate;
        this.outputSize = outputSize;
    }

    // ---------------- MOCK TRAIN ----------------

    public float[][] getFakeTrainingData(int numElements, int sizeOfData){
        float[][] data = new float[numElements][sizeOfData];
        java.util.Random rand = new java.util.Random();

        for (int i = 0; i < numElements; i++) {
            for (int k = 0; k < sizeOfData; k++) {
                data[i][k] = rand.nextFloat();
            }
        }

        return data;
    }

    public float[][] getFakeLabels(int numElements) {
        float[][] labels = new float[numElements][outputSize];
        java.util.Random rand = new java.util.Random();

        for (int i = 0; i < numElements; i++) {
            int index = rand.nextInt(outputSize);
            labels[i][index] = 1.0f;
        }

        return labels;
    }

    public void mockTrain(int numElementsToTrain, int inputSize, int epochs){
        float[][] trainingData = getFakeTrainingData(numElementsToTrain, inputSize);
        float[][] labels = getFakeLabels(numElementsToTrain);
        train(trainingData, labels, epochs);
    }

    // ---------------- REAL TRAIN ----------------

    public void trainRealData(String cleanedFilePath, String embeddingsPath, int epochs) {

        try {
            // 1. Load embeddings JSON
            Gson gson = new Gson();
            Map<String, Object> jsonData = gson.fromJson(
                    new FileReader(embeddingsPath),
                    new TypeToken<Map<String, Object>>(){}.getType()
            );

            Map<String, List<Double>> loaded = (Map<String, List<Double>>) jsonData.get("embeddings");

            embeddings = new HashMap<>();
            vocabList = new ArrayList<>();
            wordToIndex = new HashMap<>();

            // SORT KEYS FOR STABLE, CONSISTENT VOCAB ORDER
            List<String> sortedWords = new ArrayList<>(loaded.keySet());
            Collections.sort(sortedWords);

            for (String word : sortedWords) {
                List<Double> vecD = loaded.get(word);
                float[] vec = new float[128];
                for (int j = 0; j < 128; j++) {
                    vec[j] = vecD.get(j).floatValue();
                }

                embeddings.put(word, vec);
                vocabList.add(word);
            }

            for (int i = 0; i < vocabList.size(); i++) {
                wordToIndex.put(vocabList.get(i), i);
            }

            // 2. Load cleaned training data into word list
            List<String> words = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(cleanedFilePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] split = line.toLowerCase().split("[^a-zA-Z0-9']+");
                    for (String w : split) {
                        if (!w.isEmpty() && embeddings.containsKey(w)) {
                            words.add(w);
                        }
                    }
                    if (embeddings.containsKey("<END>")) {
                        words.add("<END>");
                    }
                }
            }

            if (words.size() < 11) {
                System.out.println("Not enough words for context window.");
                return;
            }

            // 3. Build context windows
            List<float[]> inputs = new ArrayList<>();
            List<float[]> labels = new ArrayList<>();

            int windowSize = 10;
            int perTokenSize = 129; // 128 embedding + 1 position
            int inputSize = windowSize * perTokenSize;

            for (int i = windowSize; i < words.size(); i++) {
                float[] inputVec = new float[inputSize];

                int offset = 0;
                for (int p = 0; p < windowSize; p++) {
                    String w = words.get(i - windowSize + p);
                    float[] emb = embeddings.get(w);

                    for (int j = 0; j < 128; j++) {
                        inputVec[offset + j] = emb[j];
                    }
                    float posNorm = (float)p / (float)(windowSize - 1);
                    inputVec[offset + 128] = posNorm;

                    offset += perTokenSize;
                }

                String nextWord = words.get(i);
                float[] label = new float[outputSize];
                Integer idx = wordToIndex.get(nextWord);
                if (idx != null && idx < outputSize) {
                    label[idx] = 1.0f;
                    inputs.add(inputVec);
                    labels.add(label);
                }
            }

            float[][] inputArr = inputs.toArray(new float[0][]);
            float[][] labelArr = labels.toArray(new float[0][]);

            train(inputArr, labelArr, epochs);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ---------------- CORE TRAIN LOOP ----------------

    public void train(float[][] trainingData, float[][] labels, int epochs) {
        System.out.println("Beginning training...");

        for (int e = 0; e < epochs; e++) {

            float epochLoss = 0f;

            for (int i = 0; i < trainingData.length; i++) {

                float[] inputData = trainingData[i];
                float[] expectedOutput = labels[i];

                float[] softmaxOutput = this.network.forward(inputData);

                float loss = ccel.calculate(softmaxOutput, expectedOutput);
                epochLoss += loss;

                float[] outputBlame = new float[softmaxOutput.length];
                for (int j = 0; j < softmaxOutput.length; j++) {
                    outputBlame[j] = softmaxOutput[j] - expectedOutput[j];
                }

                this.network.backwards(outputBlame, this.learningRate);
            }

            float avgLoss = epochLoss / trainingData.length;
            System.out.println("Epoch " + e + " - Avg Loss: " + avgLoss);

            // Save every 1000 epochs
            network.saveWeights("src/main/resources/networkWeights.json");
            // System.out.println("Checkpoint saved at epoch " + e);
//            if (e % 5 == 0 && e > 0) {
//                network.saveWeights("src/main/resources/networkWeights.json");
//                System.out.println("Checkpoint saved at epoch " + e);
//            }
        }
    }
}