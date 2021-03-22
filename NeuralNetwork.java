package main;

import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork {
	private static double LEARNINGRATE = 0.01f;
	private int n;
	private ArrayList<double[][]> weights;
	private ArrayList<double[][]> biases;
	private ArrayList<double[][]> activations;
	private ArrayList<double[][]> weightGradients;
	private ArrayList<double[][]> biasGradients;
	
	public NeuralNetwork(int... identity) {
		this.n = identity.length;
		this.weights = new ArrayList<double[][]>();
		this.biases = new ArrayList<double[][]>();
		this.activations = new ArrayList<double[][]>();
		this.weightGradients = new ArrayList<double[][]>();
		this.biasGradients = new ArrayList<double[][]>();
		//
		for(int i = 1; i < this.n; i++) {
			this.weights.add(this.randomVector(identity[i], identity[i-1]));
		}
		for(int i = 1; i < this.n; i++) {
			this.biases.add(this.randomVector(identity[i], 1));
		}
		for(int i = 0; i < this.n; i++) {
			this.activations.add(new double[identity[i]][1]);
		}
		for(int i = 1; i < this.n; i++) {
			this.weightGradients.add(new double[identity[i]][identity[i-1]]);
		}
		for(int i = 1; i < this.n; i++) {
			this.biasGradients.add(new double[identity[i]][1]);
		}
		//
		this.showState();
	}
	
	public double[][] predict(double[][] input) {
        if(input.length != this.activations.get(0).length || input[0].length != this.activations.get(0)[0].length) {
            throw new IllegalArgumentException("Prediction Error!");
        }
		this.activations.set(0, input);
		for(int i = 1; i < this.activations.size(); i++) {
			this.activations.set(i, this.sigmoid(this.add(this.multiply(this.weights.get(i-1), this.activations.get(i-1)), this.biases.get(i-1))));
		}
		return this.activations.get(this.n-1);
	}
	
	public void train(double[][] input, double[][] target) {
		//calculate activations
		this.predict(input);
		//calculate weight gradients
		for(int l = 0; l < this.weightGradients.size(); l++) {
			for(int i = 0; i < this.weightGradients.get(l).length; i++) {
				for(int j = 0; j < this.weightGradients.get(l)[0].length; j++) {
					this.weightGradients.get(l)[i][j] = this.gradientOfWeight(l, i, j, target);
				}
			}
		}
		//calculated bias gradients
		for(int l = 0; l < this.biasGradients.size(); l++) {
			for(int i = 0; i < this.biasGradients.get(l).length; i++) {
				for(int j = 0; j < this.biasGradients.get(l)[0].length; j++) {
					this.biasGradients.get(l)[i][j] = this.gradientOfBias(l, i, j, target);
				}
			}
		}
		//apply gradient
		for(int i = 0; i < this.weights.size(); i++) {
			this.weights.set(i, this.subtract(this.weights.get(i), this.weightGradients.get(i)));
		}
		for(int i = 0; i < this.biases.size(); i++) {
			this.biases.set(i, this.subtract(this.biases.get(i), this.biasGradients.get(i)));
		}
	}
	
	private double gradientOfWeight(int l, int i, int j, double[][] t) { //when referring to A, use l+1 because A[0] is input vector, n-1 because n starts at 1
		double z = this.activations.get(l)[j][0] * (1.0 - this.activations.get(l + 1)[i][0]);
		if((l + 1) < (this.n - 1)) {
			double sum = 0.0;
			for(int k = 0; k < this.weights.get(l + 1).length; k++) {
				sum += this.gradientOfWeight(l + 1, k, i, t) * this.weights.get(l + 1)[k][i];
			}
			return (z * sum);
		} else if((l + 1) == (this.n - 1)) {
			return 2.0 * (this.activations.get(l + 1)[i][0] - t[i][0]) * z * this.activations.get(l + 1)[i][0];
		}
		throw new IllegalArgumentException("Weight Gradient Calculation Error!");
	}
	
	private double gradientOfBias(int l, int i, int j, double[][] t) { //when referring to A, use l+1 because A[0] is input vector, n-1 because n starts at 1
		double z = (1.0 - this.activations.get(l + 1)[i][0]);
		if((l + 1) < (this.n - 1)) {
			double sum = 0.0;
			for(int k = 0; k < this.weights.get(l + 1).length; k++) {
				sum += this.gradientOfWeight(l + 1, k, i, t) * this.weights.get(l + 1)[k][i];
			}
			return (z * sum);
		} else if((l + 1) == (this.n - 1)) {
			return 2.0 * (this.activations.get(l + 1)[i][0] - t[i][0]) * z * this.activations.get(l + 1)[i][0];
		}
		throw new IllegalArgumentException("Bias Gradient Calculation Error!");
	}
	
	private double sigma(double x) {
		return (1.0/(1.0 + Math.exp(-x)));
	}
	
	private double normalDistribution(double x) {
		return (Math.exp(-(x*x)/2.0)/Math.sqrt(2*Math.PI));
	}
	
	private double[][] randomVector(int row, int col) {
		double[][] vector = new double[row][col];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				vector[i][j] = this.normalDistribution(-5 + 10*Math.random());
			}
		}
		return vector;
	}
	
	private double[][] multiply(double[][] A, double[][] B) {
        if(A[0].length != B.length) {
            throw new IllegalArgumentException("Multiplication Error!");
        } 
        double[][] C = new double[A.length][B[0].length];
        for(int i = 0; i < A.length; i++) {
        	for(int j = 0; j < B[0].length; j++) {
        		for(int k = 0; k < A[0].length; k++) {
        			C[i][j] += (A[i][k] * B[k][j]);
        		}
        	}
        }
        return C;
	}
	
	private double[][] sigmoid(double[][] A) {
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < A.length; i++) {
        	for(int j = 0; j < A[0].length; j++) {
        		C[i][j] = this.sigma(A[i][j]);
        	}
        }
        return C;
	}
	
	private double[][] add(double[][] A, double[][] B) {
        if(A.length != B.length || A[0].length != B[0].length) {
            throw new IllegalArgumentException("Addition Error!");
        } 
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < A.length; i++) {
        	for(int j = 0; j < A[0].length; j++) {
        		C[i][j] = (A[i][j] + B[i][j]);
        	}
        }
        return C;
	}
	
	private double[][] subtract(double[][] A, double[][] B) {
        if(A.length != B.length || A[0].length != B[0].length) {
            throw new IllegalArgumentException("Subtraction Error!");
        } 
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < A.length; i++) {
        	for(int j = 0; j < A[0].length; j++) {
        		C[i][j] = (A[i][j] - LEARNINGRATE*B[i][j]);
        	}
        }
        return C;
	}
	
	public void showState() {
		System.out.println("Weights:");
		for(double[][] a: this.weights) {
			System.out.println(Arrays.deepToString(a));
		}
		System.out.println("Biases:");
		for(double[][] a: this.biases) {
			System.out.println(Arrays.deepToString(a));
		}
		System.out.println("Activations:");
		for(double[][] a: this.activations) {
			System.out.println(Arrays.deepToString(a));
		}
		System.out.println("Weight Gradients:");
		for(double[][] a: this.weightGradients) {
			System.out.println(Arrays.deepToString(a));
		}
		System.out.println("Bias Gradients:");
		for(double[][] a: this.biasGradients) {
			System.out.println(Arrays.deepToString(a));
		}
	}
}
