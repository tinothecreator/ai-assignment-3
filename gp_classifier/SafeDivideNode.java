package gp_classifier;

public class SafeDivideNode extends DivideNode {
    public SafeDivideNode(Node left, Node right) {
        super(left, right);
    }

    @Override
    public double evaluate(double[] features) {
        double denominator = super.getRight().evaluate(features);
        return (Math.abs(denominator) < 1e-5) ? 0.0 : super.getLeft().evaluate(features) / denominator;
    }

    @Override
    public Node clone() {
        return new SafeDivideNode(super.getLeft().clone(), super.getRight().clone());
    }

    @Override
    public Node copyWith(Node left, Node right) {
        return new SafeDivideNode(left, right);
    }
}
