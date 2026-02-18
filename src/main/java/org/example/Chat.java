package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Chat {

    private final Network network;
    private final Tokenizer tokenizer;
    private final int windowSize;

    public Chat(Network network, Tokenizer tokenizer, int windowSize) {
        this.network = network;
        this.tokenizer = tokenizer;
        this.windowSize = windowSize;
    }

    public void start() {
        System.out.println("Chat started. Type 'exit' to quit.");

        Scanner scanner = new Scanner(System.in);

        // Conversation history as token indices
        List<Integer> history = new ArrayList<>();

        while (true) {
            System.out.print("You: ");
            String userInput = scanner.nextLine().trim();

            if (userInput.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye.");
                break;
            }

            // Tokenize user input
            List<Integer> userTokens = tokenizeSentence(userInput);
            history.addAll(userTokens);

            // Generate response
            String response = generateResponse(history);

            System.out.println("Bot: " + response);

            // Add bot tokens to history
            List<Integer> botTokens = tokenizeSentence(response);
            history.addAll(botTokens);

            // Trim history to window size
            if (history.size() > windowSize) {
                int excess = history.size() - windowSize;
                history = history.subList(excess, history.size());
            }
        }
    }

    // Convert sentence → list of token indices
    private List<Integer> tokenizeSentence(String sentence) {
        List<Integer> tokens = new ArrayList<>();
        String[] words = sentence.split("\\s+");

        for (String w : words) {
            int idx = tokenizer.getIndex(w);
            if (idx == -1) {
                // Unknown word → skip or map to 0
                continue;
            }
            tokens.add(idx);
        }
        return tokens;
    }

    // Generate a response using autoregressive next-token prediction
    private String generateResponse(List<Integer> history) {

        List<Integer> window = getWindow(history);

        StringBuilder sb = new StringBuilder();

        // Generate up to 20 tokens
        for (int i = 0; i < 20; i++) {

            float[] inputVector = buildInputVector(window);

            float[] output = network.forward(inputVector);

            int nextToken = argmax(output);

            String nextWord = tokenizer.getWord(nextToken);

            // Stop if model predicts end-of-sentence marker
            if (nextWord.equals("<END>")) {
                break;
            }

            sb.append(nextWord).append(" ");

            // Slide window
            window.remove(0);
            window.add(nextToken);
        }

        return sb.toString().trim();
    }

    // Extract last windowSize tokens
    private List<Integer> getWindow(List<Integer> history) {
        List<Integer> window = new ArrayList<>();

        int start = Math.max(0, history.size() - windowSize);
        for (int i = start; i < history.size(); i++) {
            window.add(history.get(i));
        }

        // Pad with zeros if needed
        while (window.size() < windowSize) {
            window.add(0);
        }

        return window;
    }

    // Build input vector: [emb1, pos1, emb2, pos2, ...]
    private float[] buildInputVector(List<Integer> window) {
        int perTokenSize = 4; // 3 embedding floats + 1 position float
        float[] vec = new float[windowSize * perTokenSize];

        for (int i = 0; i < windowSize; i++) {
            int token = window.get(i);
            String word = tokenizer.getWord(token);
            float[] emb = tokenizer.getEmbedding(word);

            int base = i * perTokenSize;

            if (emb != null && emb.length == 3) {
                vec[base] = emb[0];
                vec[base + 1] = emb[1];
                vec[base + 2] = emb[2];
            } else {
                vec[base] = 0;
                vec[base + 1] = 0;
                vec[base + 2] = 0;
            }

            // Position encoding (simple normalized index)
            vec[base + 3] = (float) i / windowSize;
        }

        return vec;
    }

    // Pick highest-probability token
    private int argmax(float[] arr) {
        int best = 0;
        float bestVal = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > bestVal) {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    }
}