package src;

import java.util.List;

import org.jblas.DoubleMatrix;

import minet.layer.Layer;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.util.Pair;

public class Agent {
	
	public static void train(Layer net, Loss loss, Optimizer optimizer, A4Dataset trainset, A4Dataset devset,
			int nEpochs, int patience) {
		int notAtPeak = 0; // the number of times not at peak
		double peakAcc = -1; // the best accuracy of the previous epochs
		double totalLoss = 0; // the total loss of the current epoch

		trainset.reset(); // reset index and shuffle the data before training

		for (int e = 0; e < nEpochs; e++) {
			totalLoss = 0;

			while (true) {
				// get the next mini-batch
				Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(trainset.getNextMiniBatch());
				if (batch == null)
					break;

				// always reset the gradients before performing backward
				optimizer.resetGradients();

				// calculate the loss value
				DoubleMatrix Yhat = net.forward(batch.first);
				double lossVal = loss.forward(batch.second, Yhat);

				// calculate gradients of the weights using backprop algorithm
				net.backward(loss.backward());

				// update the weights using the calculated gradients
				optimizer.updateWeights();

				totalLoss += lossVal;
			}

			// evaluate and print performance
			double trainAcc = eval(net, trainset);
			double valAcc = eval(net, devset);
			System.out.printf("epoch: %4d\tloss: %5.4f\ttrain-accuracy: %3.4f\tdev-accuracy: %3.4f\n", e, totalLoss,
					trainAcc, valAcc);

			// check termination condition
			if (valAcc <= peakAcc) {
				notAtPeak += 1;
				System.out.printf("not at peak %d times consecutively\n", notAtPeak);
			} else {
				notAtPeak = 0;
				peakAcc = valAcc;
			}
			if (notAtPeak == patience)
				break;
		}

		System.out.println("\ntraining is finished");
	}

	public static double eval(Layer net, A4Dataset testset) {
		// reset index of the data
		testset.reset();

		// the number of correct predictions so far
		double correct = 0;

		while (true) {
			// we evaluate per mini-batch
			Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(testset.getNextMiniBatch());
			if (batch == null)
				break;

			// perform forward pass to compute Yhat (the predictions)
			// each row of Yhat is a probabilty distribution over 10 digits
			DoubleMatrix Yhat = net.forward(batch.first);

			// the predicted digit for each image is the one with the highest probability
			int[] preds = Yhat.rowArgmaxs();

			// count how many predictions are correct
			for (int i = 0; i < preds.length; i++) {
				if (preds[i] == (int) batch.second.data[i])
					correct++;
			}
		}

		// compute classification accuracy
		double acc = correct / testset.getSize();
		return acc;
	}

	public static Pair<DoubleMatrix, DoubleMatrix> fromBatch(List<Pair<double[], Integer>> batch) {
		if (batch == null)
			return null;

		double[][] xs = new double[batch.size()][];
		double[] ys = new double[batch.size()];
		for (int i = 0; i < batch.size(); i++) {
			xs[i] = batch.get(i).first;
			ys[i] = (double) batch.get(i).second;
		}
		DoubleMatrix X = new DoubleMatrix(xs);
		DoubleMatrix Y = new DoubleMatrix(ys.length, 1, ys);
		return new Pair<DoubleMatrix, DoubleMatrix>(X, Y);
	}
}
