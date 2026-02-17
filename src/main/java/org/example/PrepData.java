package org.example;

import java.io.*;
import java.util.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class PrepData {

    public static void main(String[] args) {
        System.out.println("Preparing training data...");
        PrepData();
        System.out.println("Done.");
    }

    public static void PrepData() {

        List<String> cleanedLines = new ArrayList<>();

        // 1. Load file from src/main/resources using classloader
        InputStream is = PrepData.class.getClassLoader().getResourceAsStream("trainingData.txt");
        if (is == null) {
            System.err.println("ERROR: trainingData.txt not found in src/main/resources/");
            return;
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {

            String line;
            while ((line = br.readLine()) != null) {

                // Remove prefixes
                if (line.startsWith("Human 1: ")) {
                    line = line.substring("Human 1: ".length());
                } else if (line.startsWith("Human 2: ")) {
                    line = line.substring("Human 2: ".length());
                }

                cleanedLines.add(line);
            }

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // 2. Build vocabulary
        Set<String> vocab = new HashSet<>();

        for (String line : cleanedLines) {
            String[] words = line.toLowerCase().split("[^a-zA-Z0-9']+");
            for (String w : words) {
                if (!w.isEmpty()) {
                    vocab.add(w);
                }
            }
        }

        // 3. Assign random embeddings
        Map<String, float[]> wordEmbeddings = new HashMap<>();
        Random rand = new Random();

        for (String word : vocab) {
            float[] vec = new float[3];
            vec[0] = rand.nextFloat();
            vec[1] = rand.nextFloat();
            vec[2] = rand.nextFloat();
            wordEmbeddings.put(word, vec);
        }

        // 4. Save JSON back into src/main/resources
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        try {
            File outFile = new File("src/main/resources/wordEmbeddings.json");
            FileWriter fw = new FileWriter(outFile);
            gson.toJson(wordEmbeddings, fw);
            fw.close();
            System.out.println("Saved embeddings for " + wordEmbeddings.size() + " words.");
            System.out.println("Output: " + outFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}