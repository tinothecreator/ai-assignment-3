package gp_classifier;

public class ConstantNode implements Node {
    private final double value;

    public ConstantNode(double value) { this.value = value; }

    @Override public double evaluate(double[] features) { return value; }
    @Override public Node clone() { return new ConstantNode(value); }
    @Override public String toString() { return String.format("%.2f", value); }
    @Override public boolean isTerminal() { return true; }
    @Override public Node getLeft() { return null; }
    @Override public Node getRight() { return null; }

    @Override
    public Node copyWith(Node left, Node right) {
        throw new UnsupportedOperationException("ConstantNode is terminal and has no children");
    }
}
