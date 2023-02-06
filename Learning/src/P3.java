package src;

import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;

import minet.layer.Layer;
import minet.layer.Linear;
import minet.layer.ReLU;
import minet.layer.Sequential;
import minet.layer.Softmax;
import minet.layer.init.WeightInit;
import minet.layer.init.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;

public class P3 extends Agent{

	public P3(Random rnd, A4Dataset trainset, A4Dataset devset, A4Dataset testset, A4Dataset vocabset) {

		double learningRate = 0.1;
		int nEpochs = 500;
		int patience = 10;
		
		// create a network
		System.out.println("\nCreating network...");
		int indims = trainset.getInputDims();
		int outdims = 50;
		Sequential net = new Sequential(new Layer[] { 
				new EmbeddingBag(indims, 100, new WeightInitXavier(),vocabset), 
				new ReLU(),
				new Linear(100, 200, new WeightInitXavier()), 
				new ReLU(),
				new Linear(200, 200, new WeightInitXavier()),
                new ReLU(),
				new Linear(200, outdims, new WeightInitXavier()), 
				new Softmax() });
		CrossEntropy loss = new CrossEntropy();
		Optimizer sgd = new SGD(net, learningRate);
		System.out.println(net);
		
		// train network
		System.out.println("\nTraining...");
		train(net, loss, sgd, trainset, devset, nEpochs, patience);
		
		// perform on test set
		double testAcc = eval(net, testset);
		System.out.printf("\nTest accuracy: %.4f\n", testAcc);
	}
	
}
