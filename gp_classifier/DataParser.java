package gp_classifier;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataParser {

    public static List<double[]> parseFeatures(String path) {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] features = new double[5];
                for (int i = 0; i < 5; i++) {
                    features[i] = Double.parseDouble(values[i]);
                }
                // Normalize to [0, 1] (adjust min/max based on your data)
                double min = -2.0, max = 2.0; // Example range
                for (int i = 0; i < features.length; i++) {
                    features[i] = (features[i] - min) / (max - min);
                }
                data.add(features);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return data;
    }

    public static List<Integer> parseLabels(String path) {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                labels.add(Integer.parseInt(values[5]));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return labels;
    }
}
