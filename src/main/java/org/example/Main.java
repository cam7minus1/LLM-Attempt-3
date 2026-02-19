package org.example;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.FileReader;
import java.util.Map;

public class Main {

    public static void main(String[] args) {
        try {
            String mode = "trainchat";

            Config config = Config.load();

            File jsonFile = new File("src/main/resources/wordEmbeddings.json");
            if (!jsonFile.exists()) {
                System.err.println("ERROR: wordEmbeddings.json not found. Run PrepData first.");
                return;
            }

            Gson gson = new Gson();
            Map<String, Object> jsonData = gson.fromJson(
                    new FileReader(jsonFile),
                    new TypeToken<Map<String, Object>>(){}.getType()
            );

            int vocabSize = (int)(double) jsonData.get("numWords");
            config.vocabSize = vocabSize;
            config.save();

            System.out.println("Loaded vocab size: " + vocabSize);
            System.out.println("Embed size: " + config.embedSize);
            System.out.println("Num heads: " + config.numHeads);
            System.out.println("Num blocks: " + config.numBlocks);
            System.out.println("Window size: " + config.windowSize);
            System.out.println("Batch size: " + config.batchSize);

            String weightsPath = "src/main/resources/networkWeights.json";
            Network network = Network.loadOrCreate(
                    weightsPath,
                    config.embedSize,
                    config.vocabSize,
                    config.numHeads,
                    config.numBlocks,
                    config.ffSize
            );

            Tokenizer tokenizer = new Tokenizer("src/main/resources/wordEmbeddings.json");

            switch (mode) {
                case "train":
                    System.out.println("=== TRAIN MODE ===");
                    Trainer trainer = new Trainer(network, config.learningRate, config.vocabSize, config.batchSize);
                    trainer.trainRealData(
                            "src/main/resources/trainingData.txt",
                            "src/main/resources/wordEmbeddings.json",
                            config.epochs,
                            config
                    );
                    network.saveWeights(weightsPath);
                    break;

                case "chat":
                    System.out.println("=== CHAT MODE ===");
                    Chat chat = new Chat(network, tokenizer, config.windowSize, config.embedSize);
                    chat.start();
                    break;

                case "trainchat":
                    System.out.println("=== TRAIN + CHAT MODE ===");
                    Trainer trainer2 = new Trainer(network, config.learningRate, config.vocabSize, config.batchSize);
                    trainer2.trainRealData(
                            "src/main/resources/trainingData.txt",
                            "src/main/resources/wordEmbeddings.json",
                            config.epochs,
                            config
                    );
                    network.saveWeights(weightsPath);
                    Chat chat2 = new Chat(network, tokenizer, config.windowSize, config.embedSize);
                    chat2.start();
                    break;

                default:
                    System.out.println("Unknown mode: " + mode);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}