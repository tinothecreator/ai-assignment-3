package gp_classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Population {
    private List<Individual> individuals;
    private final int populationSize;
    private final double mutationRate;
    private final Random rand;

    // Updated constructor with Random parameter
    public Population(int size, int numFeatures, double mutationRate, Random rand) {
        this.populationSize = size;
        this.mutationRate = mutationRate;
        this.rand = rand;
        individuals = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            individuals.add(new Individual(numFeatures));
        }

        // Debug: Print first 3 individuals' outputs
        for (int i = 0; i < 3; i++) {
            Individual ind = individuals.get(i);
            double output = ind.getRoot().evaluate(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }); // Example input
            System.out.println("Initial Individual " + i + " Output: " + output);
        }
    }

    public List<Individual> getIndividuals() {
        return this.individuals;
    }

    // Tournament selection
    public Individual selectParent() {
        Individual best = null;
        for (int i = 0; i < 5; i++) {
            Individual candidate = individuals.get(rand.nextInt(populationSize));
            if (best == null || candidate.getFitness() > best.getFitness())
                best = candidate;
        }
        return best;
    }

    // Evolve the population
    public void evolve(int numFeatures) {
        List<Individual> newGeneration = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            Individual parent1 = selectParent();
            Individual parent2 = selectParent();
            Individual child = parent1.crossover(parent2);
            if (rand.nextDouble() < mutationRate)
                child = child.mutate(numFeatures);
            newGeneration.add(child);
        }
        individuals = newGeneration;
    }

    // Get the best individual
    public Individual getBest() {
        return individuals.stream()
                .max((a, b) -> Double.compare(a.getFitness(), b.getFitness()))
                .orElse(null);
    }
}
