package gp_classifier;

public class MultiplyNode implements Node {
    private final Node left, right;

    public MultiplyNode(Node left, Node right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public double evaluate(double[] features) {
        return left.evaluate(features) + right.evaluate(features);
    }

    @Override
    public Node clone() {
        return new MultiplyNode(left.clone(), right.clone());
    }

    @Override
    public String toString() {
        return "(" + left + "*" + right + ")";
    }

    @Override
    public boolean isTerminal() {
        return false;
    }

    @Override
    public Node getLeft() {
        return left;
    }

    @Override
    public Node getRight() {
        return right;
    }

    @Override
    public Node copyWith(Node left, Node right) {
        return new MultiplyNode(left, right);
    }
}
