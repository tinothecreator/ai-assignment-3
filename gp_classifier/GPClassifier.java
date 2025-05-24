package gp_classifier;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GPClassifier {

    private static void savePredictions(List<double[]> testData, List<Integer> testLabels,
            Individual best, String path, List<Integer> predictedOut) {
        try (FileWriter writer = new FileWriter(path)) {
            writer.write("Predicted,Actual\n"); // Header
            for (int i = 0; i < testData.size(); i++) {
                double output = best.getRoot().evaluate(testData.get(i));
                int predicted = (output >= 0.5) ? 1 : 0;
                predictedOut.add(predicted);
                writer.write(predicted + "," + testLabels.get(i) + "\n");
            }
            System.out.println("Predictions saved to " + path);
        } catch (IOException e) {
            System.err.println("Error writing predictions: " + e.getMessage());
        }
    }

    private static double calculateF1(List<Integer> actual, List<Integer> predicted) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < actual.size(); i++) {
            int a = actual.get(i);
            int p = predicted.get(i);
            if (p == 1 && a == 1) tp++;
            if (p == 1 && a == 0) fp++;
            if (p == 0 && a == 1) fn++;
        }
        double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
        double recall = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
        return (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
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
        int numFeatures = 5;
        Population population = new Population(1000, numFeatures, 0.15, rand);

        // Validation setup
        List<double[]> validationData = trainData.subList(0, trainData.size() / 5);
        List<Integer> validationLabels = trainLabels.subList(0, trainLabels.size() / 5);

        double bestValAccuracy = 0;
        int patience = 5;
        int noImprovementEpochs = 0;

        // Training loop with progress reporting
        for (int gen = 0; gen < 100; gen++) {
            // Evaluate current population
            population.getIndividuals().parallelStream().forEach(ind ->
                    ind.calculateFitness(trainData, trainLabels));

            // Get best individual
            Individual best = population.getBest();

            // Calculate validation accuracy
            best.calculateFitness(validationData, validationLabels);
            double valAcc = best.getFitness();

            // Print generation statistics
            System.out.printf("Generation %02d: Train=%.2f Val=%.2f%n",
                    gen, best.getFitness(), valAcc);

            // Early stopping check
            if (valAcc > bestValAccuracy) {
                bestValAccuracy = valAcc;
                noImprovementEpochs = 0;
            } else {
                noImprovementEpochs++;
            }

            if (noImprovementEpochs >= patience) {
                System.out.println("Early stopping at generation " + gen);
                break;
            }

            // Evolve population
            population.evolve(numFeatures);
        }

        // Evaluate on train set
        Individual best = population.getBest();
        List<Integer> trainPredictions = new ArrayList<>();
        for (double[] data : trainData) {
            double out = best.getRoot().evaluate(data);
            trainPredictions.add(out >= 0.5 ? 1 : 0);
        }
        double trainAcc = best.getFitness();
        double trainF1 = calculateF1(trainLabels, trainPredictions);

        // Evaluate on test set
        List<Integer> testPredictions = new ArrayList<>();
        savePredictions(testData, testLabels, best, "./results/gp_predictions.csv", testPredictions);
        best.calculateFitness(testData, testLabels);
        double testAcc = best.getFitness();
        double testF1 = calculateF1(testLabels, testPredictions);

        // Print final metrics
        System.out.printf("Final Train Accuracy: %.2f%%%n", trainAcc * 100);
        System.out.printf("Final Train F1 Score: %.2f%n", trainF1);
        System.out.printf("Final Test Accuracy: %.2f%%%n", testAcc * 100);
        System.out.printf("Final Test F1 Score: %.2f%n", testF1);
        System.out.println("Best Model: " + best.getRoot());

        // Print table summary
        System.out.println("\n=== Final Results ===");
        System.out.printf("| Model | Acc (Train) | F1 (Train) | Acc (Test) | F1 (Test) |%n");
        System.out.printf("| GP    | %.2f       | %.2f       | %.2f       | %.2f     |%n",
                trainAcc, trainF1, testAcc, testF1);
    }
}
