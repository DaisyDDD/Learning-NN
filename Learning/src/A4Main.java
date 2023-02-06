package src;

import org.jblas.util.Logger;

import minet.example.mnist.MNISTDataset;
import minet.layer.Layer;
import minet.layer.Linear;
import minet.layer.ReLU;
import minet.layer.Sequential;
import minet.layer.Softmax;
import minet.layer.init.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.optim.Optimizer;
import minet.optim.SGD;

import java.io.IOException;
import java.util.Random;
import src.EmbeddingBag;

public class A4Main {

	/**
	 * Example A4Main class. Feel free to edit this file
	 * 
	 */
	public static void main(String[] args) throws IOException {
		long startTime=System.currentTimeMillis();
		if (args.length < 6) {
			System.out.println(
					"Usage: java A4Main <part1/part2/part3/part4> <seed> <trainFile> <devFile> <testFile> <vocabFile> <classesFile>");
			return;
		}

		// set jblas random seed (for reproducibility)
		int seed = Integer.parseInt(args[1]);
		org.jblas.util.Random.seed(seed);
		Random rnd = new Random(seed);

		// turn off jblas info messages
		Logger.getLogger().setLevel(Logger.WARNING);
		int batchsize = 50;

		// load datasets
		System.out.println("\nLoading data...");
		A4Dataset trainset = new A4Dataset(batchsize, true, rnd);
		A4Dataset devset = new A4Dataset(batchsize, false, rnd);
		A4Dataset testset = new A4Dataset(batchsize, false, rnd);
		// set input dimentionss
		if (args[0].equals("part3")) {
			trainset.setInputDims(8595);
			devset.setInputDims(8595);
			testset.setInputDims(8595);
		} else {
			trainset.setInputDims(3250);
			devset.setInputDims(3250);
			testset.setInputDims(3250);
		}
		
		
		try {
			//Read file and transfer to one-hot-encoding format
			trainset.fromFile(args[2]);
			devset.fromFile(args[3]);
			testset.fromFile(args[4]);

		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.printf("train: %d instances\n", trainset.getSize());
		System.out.printf("dev: %d instances\n", devset.getSize());
		System.out.printf("test: %d instances\n", testset.getSize());

		switch (args[0]) {
		case "part1": {
			new P1(rnd, trainset, devset, testset);
			break;
		}
		case "part2": {
			new P2(rnd, trainset, devset, testset);
			break;
		}
		case "part3": {
			A4Dataset vocabset = new A4Dataset(batchsize, true, rnd);
			vocabset.readVocab(args[5]);
			new P3(rnd, trainset, devset, testset, vocabset);
		}
		}
		long endTime=System.currentTimeMillis(); 
		System.out.println("Total running time: "+(endTime-startTime)+"ms");  
	}
}
