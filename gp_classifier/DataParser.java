package gp_classifier;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataParser {
    private static double[] featureMin = new double[5];
    private static double[] featureMax = new double[5];
    private static boolean minMaxCalculated = false;

    public static List<double[]> parseFeatures(String path) {
        List<double[]> data = new ArrayList<>();
        
        // First pass: Calculate min/max if not already done
        if (!minMaxCalculated) {
            calculateMinMax(path);
            minMaxCalculated = true;
        }

        // Second pass: Normalize data
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] features = new double[5];
                
                // Parse and normalize each feature
                for (int i = 0; i < 5; i++) {
                    double value = Double.parseDouble(values[i]);
                    // Handle constant features (min == max)
                    if (featureMax[i] == featureMin[i]) {
                        features[i] = 0.5; // Midpoint value
                    } else {
                        features[i] = (value - featureMin[i]) / (featureMax[i] - featureMin[i]);
                    }
                }
                data.add(features);
            }
        } catch (Exception e) {
            System.err.println("Error reading features: " + e.getMessage());
            e.printStackTrace();
        }
        return data;
    }

    private static void calculateMinMax(String path) {
        // Initialize with extreme values
        for (int i = 0; i < 5; i++) {
            featureMin[i] = Double.MAX_VALUE;
            featureMax[i] = -Double.MAX_VALUE;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                for (int i = 0; i < 5; i++) {
                    double value = Double.parseDouble(values[i]);
                    if (value < featureMin[i]) featureMin[i] = value;
                    if (value > featureMax[i]) featureMax[i] = value;
                }
            }
        } catch (Exception e) {
            System.err.println("Error calculating min/max: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static List<Integer> parseLabels(String path) {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                // Ensure label is either 0 or 1
                int label = Integer.parseInt(values[5]);
                labels.add(label > 0 ? 1 : 0); // Force binary classification
            }
        } catch (Exception e) {
            System.err.println("Error reading labels: " + e.getMessage());
            e.printStackTrace();
        }
        return labels;
    }

    // Optional: Print min/max values for debugging
    public static void printMinMax() {
        System.out.println("Feature Normalization Ranges:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("Feature %d: Min=%.4f, Max=%.4f%n", 
                i, featureMin[i], featureMax[i]);
        }
    }
}