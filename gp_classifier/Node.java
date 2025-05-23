package gp_classifier;

public interface Node {
    double evaluate(double[] features);
    Node clone();
    String toString();
    boolean isTerminal();
    Node getLeft();
    Node getRight();
    // Only non-terminal nodes need to implement this
    default Node copyWith(Node left, Node right) {
        throw new UnsupportedOperationException("copyWith not supported for terminal nodes");
    }

    
}
