package desicion_tree;

import java.util.*;
import java.io.*;

/**
 * Decision Tree implementation for BTC prediction using CSV data
 * Handles continuous financial data by discretizing into categorical ranges
 */
public class BTCDecisionTreePredictor {
    
    // Decision Tree Node class
    static class DecisionNode {
        String attribute;
        String classification;
        Map<String, DecisionNode> children;
        boolean isLeaf;
        
        public DecisionNode() {
            this.children = new HashMap<>();
            this.isLeaf = false;
        }
        
        public DecisionNode(String classification) {
            this.classification = classification;
            this.isLeaf = true;
            this.children = new HashMap<>();
        }
    }
    
    // Training instance class
    static class Instance {
        Map<String, String> attributes;
        String classification;
        
        public Instance(Map<String, String> attributes, String classification) {
            this.attributes = new HashMap<>(attributes);
            this.classification = classification;
        }
    }
    
    // Raw data point for discretization
    static class RawDataPoint {
        double open, high, low, close, adjClose;
        int output;
        
        public RawDataPoint(double open, double high, double low, double close, double adjClose, int output) {
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
            this.adjClose = adjClose;
            this.output = output;
        }
    }
    
    private DecisionNode root;
    private List<String> attributeNames;
    private Map<String, double[]> discretizationThresholds;
    
    public BTCDecisionTreePredictor() {
        this.discretizationThresholds = new HashMap<>();
    }
    
    /**
     * Load and parse CSV file
     */
    public List<RawDataPoint> loadCSV(String filename) {
        List<RawDataPoint> data = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // Skip header
            
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length >= 6) {
                    try {
                        double open = Double.parseDouble(values[0]);
                        double high = Double.parseDouble(values[1]);
                        double low = Double.parseDouble(values[2]);
                        double close = Double.parseDouble(values[3]);
                        double adjClose = Double.parseDouble(values[4]);
                        int output = Integer.parseInt(values[5]);
                        
                        data.add(new RawDataPoint(open, high, low, close, adjClose, output));
                    } catch (NumberFormatException e) {
                        System.err.println("Skipping invalid line: " + line);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        
        return data;
    }
    
    /**
     * Calculate discretization thresholds using quartiles
     */
    private void calculateDiscretizationThresholds(List<RawDataPoint> data) {
        List<Double> openValues = new ArrayList<>();
        List<Double> highValues = new ArrayList<>();
        List<Double> lowValues = new ArrayList<>();
        List<Double> closeValues = new ArrayList<>();
        List<Double> adjCloseValues = new ArrayList<>();
        
        for (RawDataPoint point : data) {
            openValues.add(point.open);
            highValues.add(point.high);
            lowValues.add(point.low);
            closeValues.add(point.close);
            adjCloseValues.add(point.adjClose);
        }
        
        discretizationThresholds.put("Open", calculateQuartiles(openValues));
        discretizationThresholds.put("High", calculateQuartiles(highValues));
        discretizationThresholds.put("Low", calculateQuartiles(lowValues));
        discretizationThresholds.put("Close", calculateQuartiles(closeValues));
        discretizationThresholds.put("AdjClose", calculateQuartiles(adjCloseValues));
    }
    
    /**
     * Calculate quartiles for discretization
     */
    private double[] calculateQuartiles(List<Double> values) {
        Collections.sort(values);
        int n = values.size();
        
        double q1 = values.get(n / 4);
        double q2 = values.get(n / 2); // median
        double q3 = values.get(3 * n / 4);
        
        return new double[]{q1, q2, q3};
    }
    
    /**
     * Discretize continuous value into categorical range
     */
    private String discretizeValue(String attribute, double value) {
        if (!discretizationThresholds.containsKey(attribute)) {
            return "Medium"; // Default fallback
        }
        
        double[] thresholds = discretizationThresholds.get(attribute);
        
        if (value <= thresholds[0]) {
            return "VeryLow";
        } else if (value <= thresholds[1]) {
            return "Low";
        } else if (value <= thresholds[2]) {
            return "Medium";
        } else {
            return "High";
        }
    }
    
    /**
     * Convert raw data points to instances with discretized attributes
     */
    private List<Instance> discretizeData(List<RawDataPoint> rawData) {
        List<Instance> instances = new ArrayList<>();
        
        for (RawDataPoint point : rawData) {
            Map<String, String> attributes = new HashMap<>();
            
            attributes.put("Open", discretizeValue("Open", point.open));
            attributes.put("High", discretizeValue("High", point.high));
            attributes.put("Low", discretizeValue("Low", point.low));
            attributes.put("Close", discretizeValue("Close", point.close));
            attributes.put("AdjClose", discretizeValue("AdjClose", point.adjClose));
            
            // Add derived features
            attributes.put("PriceChange", point.close > point.open ? "Positive" : "Negative");
            attributes.put("VolatilityRange", (point.high - point.low) > 0.05 ? "High" : "Low");
            
            String classification = String.valueOf(point.output);
            instances.add(new Instance(attributes, classification));
        }
        
        return instances;
    }
    
    /**
     * Train the decision tree using ID3 algorithm
     */
    public void train(String trainFile) {
        System.out.println("Loading training data from: " + trainFile);
        List<RawDataPoint> rawTrainingData = loadCSV(trainFile);
        System.out.println("Loaded " + rawTrainingData.size() + " training samples");
        
        // Calculate discretization thresholds from training data
        calculateDiscretizationThresholds(rawTrainingData);
        
        // Convert to discretized instances
        List<Instance> trainingData = discretizeData(rawTrainingData);
        
        // Define attributes
        this.attributeNames = Arrays.asList("Open", "High", "Low", "Close", "AdjClose", "PriceChange", "VolatilityRange");
        
        // Train the tree
        this.root = id3(trainingData, new HashSet<>(attributeNames));
        
        System.out.println("Training completed!");
        
        // Show class distribution
        showClassDistribution(trainingData);
    }
    
    /**
     * Test the decision tree
     */
    public void test(String testFile) {
        System.out.println("\nLoading test data from: " + testFile);
        List<RawDataPoint> rawTestData = loadCSV(testFile);
        System.out.println("Loaded " + rawTestData.size() + " test samples");
        
        // Convert to discretized instances using training thresholds
        List<Instance> testData = discretizeData(rawTestData);
        
        // Calculate accuracy
        double accuracy = evaluateAccuracy(testData);
        System.out.println("Test Accuracy: " + String.format("%.2f", accuracy * 100) + "%");
        
        // Calculate detailed metrics
        calculateDetailedMetrics(testData);
        
        // Show some sample predictions
        showSamplePredictions(testData, 10);
    }
    
    /**
     * ID3 Algorithm implementation
     */
    private DecisionNode id3(List<Instance> examples, Set<String> attributes) {
        // If all examples have the same classification
        if (allSameClassification(examples)) {
            return new DecisionNode(examples.get(0).classification);
        }
        
        // If attribute set is empty
        if (attributes.isEmpty()) {
            String mostCommonClass = getMostCommonClass(examples);
            return new DecisionNode(mostCommonClass);
        }
        
        // Find best attribute
        String bestAttribute = getBestAttribute(examples, attributes);
        
        // Create decision node
        DecisionNode node = new DecisionNode();
        node.attribute = bestAttribute;
        
        // Remove best attribute from attribute set
        Set<String> remainingAttributes = new HashSet<>(attributes);
        remainingAttributes.remove(bestAttribute);
        
        // Get all possible values for the best attribute
        Set<String> possibleValues = getPossibleValues(examples, bestAttribute);
        
        // For each possible value of best attribute
        for (String value : possibleValues) {
            List<Instance> subset = getSubsetWithAttributeValue(examples, bestAttribute, value);
            
            if (subset.isEmpty()) {
                String mostCommonClass = getMostCommonClass(examples);
                node.children.put(value, new DecisionNode(mostCommonClass));
            } else {
                node.children.put(value, id3(subset, remainingAttributes));
            }
        }
        
        return node;
    }
    
    /**
     * Check if all examples have the same classification
     */
    private boolean allSameClassification(List<Instance> examples) {
        if (examples.isEmpty()) return true;
        
        String firstClass = examples.get(0).classification;
        for (Instance instance : examples) {
            if (!instance.classification.equals(firstClass)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Get the most common classification
     */
    private String getMostCommonClass(List<Instance> examples) {
        Map<String, Integer> classCounts = new HashMap<>();
        
        for (Instance instance : examples) {
            classCounts.put(instance.classification, 
                classCounts.getOrDefault(instance.classification, 0) + 1);
        }
        
        return classCounts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("0");
    }
    
    /**
     * Find the best attribute using Information Gain
     */
    private String getBestAttribute(List<Instance> examples, Set<String> attributes) {
        double maxGain = -1;
        String bestAttribute = null;
        
        double baseEntropy = calculateEntropy(examples);
        
        for (String attribute : attributes) {
            double gain = calculateInformationGain(examples, attribute, baseEntropy);
            if (gain > maxGain) {
                maxGain = gain;
                bestAttribute = attribute;
            }
        }
        
        return bestAttribute != null ? bestAttribute : attributes.iterator().next();
    }
    
    /**
     * Calculate entropy
     */
    private double calculateEntropy(List<Instance> examples) {
        if (examples.isEmpty()) return 0;
        
        Map<String, Integer> classCounts = new HashMap<>();
        for (Instance instance : examples) {
            classCounts.put(instance.classification, 
                classCounts.getOrDefault(instance.classification, 0) + 1);
        }
        
        double entropy = 0.0;
        int totalSize = examples.size();
        
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = (double) count / totalSize;
                entropy -= probability * (Math.log(probability) / Math.log(2));
            }
        }
        
        return entropy;
    }
    
    /**
     * Calculate information gain
     */
    private double calculateInformationGain(List<Instance> examples, String attribute, double baseEntropy) {
        Map<String, List<Instance>> partitions = new HashMap<>();
        
        for (Instance instance : examples) {
            String value = instance.attributes.get(attribute);
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(instance);
        }
        
        double weightedEntropy = 0.0;
        int totalSize = examples.size();
        
        for (List<Instance> partition : partitions.values()) {
            double weight = (double) partition.size() / totalSize;
            weightedEntropy += weight * calculateEntropy(partition);
        }
        
        return baseEntropy - weightedEntropy;
    }
    
    /**
     * Get all possible values for an attribute
     */
    private Set<String> getPossibleValues(List<Instance> examples, String attribute) {
        Set<String> values = new HashSet<>();
        for (Instance instance : examples) {
            values.add(instance.attributes.get(attribute));
        }
        return values;
    }
    
    /**
     * Get subset with specific attribute value
     */
    private List<Instance> getSubsetWithAttributeValue(List<Instance> examples, String attribute, String value) {
        List<Instance> subset = new ArrayList<>();
        for (Instance instance : examples) {
            if (value.equals(instance.attributes.get(attribute))) {
                subset.add(instance);
            }
        }
        return subset;
    }
    
    /**
     * Predict classification for new instance
     */
    public String predict(Map<String, String> instance) {
        return predictRecursive(root, instance);
    }
    
    private String predictRecursive(DecisionNode node, Map<String, String> instance) {
        if (node.isLeaf) {
            return node.classification;
        }
        
        String attributeValue = instance.get(node.attribute);
        DecisionNode child = node.children.get(attributeValue);
        
        if (child == null) {
            return "0"; // Default prediction
        }
        
        return predictRecursive(child, instance);
    }
    
    /**
     * Evaluate accuracy
     */
    public double evaluateAccuracy(List<Instance> testData) {
        int correct = 0;
        int total = testData.size();
        
        for (Instance instance : testData) {
            String predicted = predict(instance.attributes);
            if (predicted.equals(instance.classification)) {
                correct++;
            }
        }
        
        return (double) correct / total;
    }
    
    /**
     * Show class distribution
     */
    private void showClassDistribution(List<Instance> data) {
        Map<String, Integer> distribution = new HashMap<>();
        for (Instance instance : data) {
            distribution.put(instance.classification, 
                distribution.getOrDefault(instance.classification, 0) + 1);
        }
        
        System.out.println("\nClass Distribution:");
        for (Map.Entry<String, Integer> entry : distribution.entrySet()) {
            double percentage = (double) entry.getValue() / data.size() * 100;
            System.out.println("Class " + entry.getKey() + ": " + entry.getValue() + " (" + 
                             String.format("%.1f", percentage) + "%)");
        }
    }
    
    /**
     * Calculate detailed performance metrics
     */
    private void calculateDetailedMetrics(List<Instance> testData) {
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        Set<String> classes = new HashSet<>();
        
        // Initialize confusion matrix
        for (Instance instance : testData) {
            classes.add(instance.classification);
        }
        
        for (String actual : classes) {
            confusionMatrix.put(actual, new HashMap<>());
            for (String predicted : classes) {
                confusionMatrix.get(actual).put(predicted, 0);
            }
        }
        
        // Fill confusion matrix
        for (Instance instance : testData) {
            String predicted = predict(instance.attributes);
            String actual = instance.classification;
            
            if (!confusionMatrix.containsKey(actual)) {
                confusionMatrix.put(actual, new HashMap<>());
            }
            confusionMatrix.get(actual).put(predicted, 
                confusionMatrix.get(actual).getOrDefault(predicted, 0) + 1);
        }
        
        // Print confusion matrix
        System.out.println("\nConfusion Matrix:");
        System.out.print("Actual\\Predicted\t");
        for (String cls : classes) {
            System.out.print(cls + "\t");
        }
        System.out.println();
        
        for (String actual : classes) {
            System.out.print(actual + "\t\t\t");
            for (String predicted : classes) {
                System.out.print(confusionMatrix.get(actual).getOrDefault(predicted, 0) + "\t");
            }
            System.out.println();
        }
        
        // Calculate per-class metrics
        System.out.println("\nPer-Class Metrics:");
        for (String className : classes) {
            int tp = confusionMatrix.get(className).getOrDefault(className, 0);
            int fp = 0, fn = 0;
            
            for (String other : classes) {
                if (!other.equals(className)) {
                    fp += confusionMatrix.get(other).getOrDefault(className, 0);
                    fn += confusionMatrix.get(className).getOrDefault(other, 0);
                }
            }
            
            double precision = tp > 0 ? (double) tp / (tp + fp) : 0;
            double recall = tp > 0 ? (double) tp / (tp + fn) : 0;
            double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
            
            System.out.println("Class " + className + " - Precision: " + String.format("%.3f", precision) + 
                             ", Recall: " + String.format("%.3f", recall) + 
                             ", F1-Score: " + String.format("%.3f", f1));
        }
    }
    
    /**
     * Show sample predictions
     */
    private void showSamplePredictions(List<Instance> testData, int numSamples) {
        System.out.println("\nSample Predictions:");
        System.out.println("Actual\tPredicted\tMatch");
        
        for (int i = 0; i < Math.min(numSamples, testData.size()); i++) {
            Instance instance = testData.get(i);
            String predicted = predict(instance.attributes);
            String actual = instance.classification;
            boolean match = predicted.equals(actual);
            
            System.out.println(actual + "\t" + predicted + "\t\t" + (match ? "YES" : "NO"));
        }
    }
    
    /**
     * Print tree structure
     */
    public void printTree() {
        System.out.println("\nDecision Tree Structure:");
        printTreeRecursive(root, "", "");
    }
    
    private void printTreeRecursive(DecisionNode node, String prefix, String attributeValue) {
        if (node.isLeaf) {
            System.out.println(prefix + attributeValue + " -> Class " + node.classification);
            return;
        }
        
        if (!attributeValue.isEmpty()) {
            System.out.println(prefix + attributeValue + " -> " + node.attribute);
        } else {
            System.out.println("Root: " + node.attribute);
        }
        
        for (Map.Entry<String, DecisionNode> entry : node.children.entrySet()) {
            printTreeRecursive(entry.getValue(), prefix + "  ", entry.getKey());
        }
    }
    
    public static void main(String[] args) {
        BTCDecisionTreePredictor predictor = new BTCDecisionTreePredictor();
        
        // Train the model
        predictor.train("data/BTC_train.csv");
        
        // Print the tree structure
        predictor.printTree();
        
        // Test the model
        predictor.test("data/BTC_test.csv");
    }
}