package org.example;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.FileReader;
import java.util.*;

public class Tokenizer {

    private final Map<String, Integer> wordToIndex = new HashMap<>();
    private final Map<Integer, String> indexToWord = new HashMap<>();
    private final Map<String, float[]> embeddings = new HashMap<>();

    public Tokenizer(String jsonPath) throws Exception {

        Gson gson = new Gson();
        Map<String, Object> jsonData = gson.fromJson(
                new FileReader(jsonPath),
                new TypeToken<Map<String, Object>>(){}.getType()
        );

        Map<String, Object> embMap = (Map<String, Object>) jsonData.get("embeddings");

        // SORT KEYS FOR STABLE, CONSISTENT VOCAB ORDER
        List<String> sortedWords = new ArrayList<>(embMap.keySet());
        Collections.sort(sortedWords);

        int index = 0;
        for (String word : sortedWords) {

            // Convert List<Double> → float[]
            List<Double> list = (List<Double>) embMap.get(word);
            float[] vec = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                vec[i] = list.get(i).floatValue();
            }

            embeddings.put(word, vec);
            wordToIndex.put(word, index);
            indexToWord.put(index, word);
            index++;
        }

        System.out.println("Tokenizer loaded " + sortedWords.size() + " words.");
    }

    public int getIndex(String word) {
        return wordToIndex.getOrDefault(word, -1);
    }

    public String getWord(int index) {
        return indexToWord.getOrDefault(index, "<UNK>");
    }

    public float[] getEmbedding(String word) {
        return embeddings.get(word);
    }
}