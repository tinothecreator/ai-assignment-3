package gp_classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Individual {
    private Node root;
    private double fitness;
    private static final int MAX_DEPTH = 4; // Increased slightly but controlled
    private static final int MAX_NODES = 100; // Add node count limit
    private static Random rand = new Random(); // Changed to non-final

    // Add static method to set Random
    public static void setRandom(Random r) {
        rand = r;
    }

    // Initialize with a random tree
    public Individual(int numFeatures) {
        this.root = buildTree(0, numFeatures, rand.nextBoolean());
    }

    // New constructor for crossover/mutation
    public Individual(Node root) {
        this.root = root;
    }

    private Node buildTree(int depth, int numFeatures, boolean grow) {
        if (depth >= MAX_DEPTH || countNodes(root) > MAX_NODES || (grow && rand.nextBoolean())) {
            // Force terminal if limits exceeded
            return rand.nextBoolean() ? new FeatureNode(rand.nextInt(numFeatures))
                    : new ConstantNode(rand.nextDouble() * 2 - 1);
        } else {
            Node left = buildTree(depth + 1, numFeatures, grow);
            Node right = buildTree(depth + 1, numFeatures, grow);
            switch (rand.nextInt(4)) {
                case 0:
                    return new AddNode(left, right);
                case 1:
                    return new SubtractNode(left, right);
                case 2:
                    return new MultiplyNode(left, right);
                default:
                    return new SafeDivideNode(left, right); // Use SafeDivideNode
            }
        }
    }

    // Crossover: Swap subtrees with another individual
    public Individual crossover(Individual other) {
        Node thisSubtree = getRandomSubtree(this.root);
        Node otherSubtree = getRandomSubtree(other.root);
        return new Individual(replaceSubtree(this.root, thisSubtree, otherSubtree));
    }

    // Mutation: Replace a random subtree
    public Individual mutate(int numFeatures) {
        Node subtree = getRandomSubtree(this.root);
        Node newSubtree = buildTree(0, numFeatures, rand.nextBoolean());
        return new Individual(replaceSubtree(this.root, subtree, newSubtree));
    }

    // Helper: Replace a subtree in the tree
    private Node replaceSubtree(Node root, Node target, Node replacement) {
        if (root == target) {
            return replacement.clone();
        }
        if (root.isTerminal()) {
            return root.clone();
        }
        Node newLeft = replaceSubtree(root.getLeft(), target, replacement);
        Node newRight = replaceSubtree(root.getRight(), target, replacement);
        return root.copyWith(newLeft, newRight);
    }

    // Helper: Get a random subtree
    private Node getRandomSubtree(Node root) {
        List<Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        return nodes.get(rand.nextInt(nodes.size()));
    }

    // Helper: Collect all nodes in the tree
    private void collectNodes(Node node, List<Node> nodes) {
        nodes.add(node);
        if (!node.isTerminal()) {
            collectNodes(node.getLeft(), nodes);
            collectNodes(node.getRight(), nodes);
        }
    }

    // Calculate fitness (accuracy)
    public void calculateFitness(List<double[]> data, List<Integer> labels) {
        int correct = 0;
        for (int i = 0; i < data.size(); i++) {
            double rawOutput = root.evaluate(data.get(i));
            // Apply sigmoid activation
            double sigmoid = 1 / (1 + Math.exp(-rawOutput));
            int predicted = (sigmoid >= 0.5) ? 1 : 0;
            if (predicted == labels.get(i))
                correct++;
        }
        fitness = (double) correct / data.size();
        // Add parsimony pressure (penalize tree size)
        int treeSize = countNodes(root);
        fitness *= Math.pow(0.99, treeSize / 50.0); // Adjust 0.99 and 50 as needed
    }

    private int countNodes(Node node) {
        if (node == null)
            return 0;
        return 1 + countNodes(node.getLeft()) + countNodes(node.getRight());
    }

    // Getters
    public double getFitness() {
        return fitness;
    }

    public Node getRoot() {
        return root;
    }
}
