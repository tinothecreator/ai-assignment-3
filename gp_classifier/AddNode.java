package gp_classifier;

public class AddNode implements Node {
    private final Node left, right;

    public AddNode(Node left, Node right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public double evaluate(double[] features) {
        return left.evaluate(features) + right.evaluate(features);
    }

    @Override
    public Node clone() {
        return new AddNode(left.clone(), right.clone());
    }

    @Override
    public String toString() {
        return "(" + left + "+" + right + ")";
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
        return new AddNode(left, right);
    }
}
