package gp_classifier;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class GPClassifier {

    private static void savePredictions(List<double[]> testData, List<Integer> testLabels,
            Individual best, String path) {
        try (FileWriter writer = new FileWriter(path)) {
            writer.write("Predicted,Actual\n"); // Header
            for (int i = 0; i < testData.size(); i++) {
                double output = best.getRoot().evaluate(testData.get(i));
                int predicted = (output >= 0.5) ? 1 : 0;
                writer.write(predicted + "," + testLabels.get(i) + "\n");
            }
            System.out.println("Predictions saved to " + path);
        } catch (IOException e) {
            System.err.println("Error writing predictions: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {
        // Parse command-line arguments
        long seed = Long.parseLong(args[0]);
        String trainPath = args[1];
        String testPath = args[2];

        // Initialize random number generator with seed
        Random rand = new Random(seed);
        Individual.setRandom(rand); // Pass to Individual class for reproducibility

        // Load data
        List<double[]> trainData = DataParser.parseFeatures(trainPath);
        List<Integer> trainLabels = DataParser.parseLabels(trainPath);
        List<double[]> testData = DataParser.parseFeatures(testPath);
        List<Integer> testLabels = DataParser.parseLabels(testPath);

        // Initialize GP
        int numFeatures = 5; // Open, High, Low, Close, Adj Close
        Population population = new Population(1000, numFeatures, 0.15, rand);

        // Updated GPClassifier.java (partial)
        List<double[]> validationData = trainData.subList(0, trainData.size() / 5); // 20% as validation
        List<Integer> validationLabels = trainLabels.subList(0, trainLabels.size() / 5);

        double bestValAccuracy = 0;
        int patience = 5;
        int noImprovementEpochs = 0;

        for (int gen = 0; gen < 100; gen++) { // Increase max generations
            population.getIndividuals().forEach(ind -> ind.calculateFitness(trainData, trainLabels));

            Individual best = population.getBest();
            best.calculateFitness(validationData, validationLabels);
            double valAcc = best.getFitness();

            if (valAcc > bestValAccuracy) {
                bestValAccuracy = valAcc;
                noImprovementEpochs = 0;
            } else {
                noImprovementEpochs++;
            }

            if (noImprovementEpochs >= patience)
                break;

            population.evolve(numFeatures);
        }

        // Evaluate on test set
        Individual best = population.getBest();
        best.calculateFitness(testData, testLabels);
        System.out.printf("Final Test Accuracy: %.2f%%%n", best.getFitness() * 100);
        System.out.println("Best Model: " + best.getRoot());

        // Save predictions
        savePredictions(testData, testLabels, best, "./results/gp_predictions.csv");
    }
}