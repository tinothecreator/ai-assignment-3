package gp_classifier;

public class SubtractNode implements Node {
    private final Node left, right;

    public SubtractNode(Node left, Node right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public double evaluate(double[] features) {
        return left.evaluate(features) + right.evaluate(features);
    }

    @Override
    public Node clone() {
        return new SubtractNode(left.clone(), right.clone());
    }

    @Override
    public String toString() {
        return "(" + left + "-" + right + ")";
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

    public Node copyWith(Node left, Node right) {
        return new SubtractNode(left, right);
    }
}
