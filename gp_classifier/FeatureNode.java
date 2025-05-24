package gp_classifier;

public class FeatureNode implements Node {
    private final int index;

    public FeatureNode(int index) { this.index = index; }

    @Override public double evaluate(double[] features) { return features[index]; }
    @Override public Node clone() { return new FeatureNode(index); }
    @Override public String toString() { return "x" + index; }
    @Override public boolean isTerminal() { return true; }
    @Override public Node getLeft() { return null; }
    @Override public Node getRight() { return null; }

    @Override
    public Node copyWith(Node left, Node right) {
        throw new UnsupportedOperationException("FeatureNode is terminal and has no children");
    }
}
