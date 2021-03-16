package main;

import java.util.Arrays;

public class Main {

	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(new int[] {2,3,2});
//		System.out.println("Prediction:");
//		System.out.println(Arrays.deepToString(nn.predict(new double[][] {{1},{2}})));
		nn.train(new double[][] {{1},{2}}, new double[][] {{1},{0}});
		nn.showState();
		nn.train(new double[][] {{4},{8}}, new double[][] {{1},{0}});
		nn.showState();
	}

}
