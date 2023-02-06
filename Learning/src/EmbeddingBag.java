package src;

import org.jblas.*;

import java.util.ArrayList;
import java.util.List;
import minet.layer.init.*;
import minet.layer.Layer;

/**
 * A class for Embedding bag layers. Feel free to modify this class for your
 * implementation.
 */
public class EmbeddingBag implements Layer, java.io.Serializable {

	private static final long serialVersionUID = -10445336293457309L;
	A4Dataset vocabset=null;
	DoubleMatrix W; // weight matrix (for simplicity, we can ignore the bias term b)
	int outdims;
	// for backward
	List<int[]> X; // store input X for computing backward, each element in this list is a sample
					// (an array of word indices).
	DoubleMatrix gW; // gradient of W

	/**
	 * Constructor for EmbeddingBag
	 * 
	 * @param vocabSize (int) vocabulary size
	 * @param outdims   (int) output of this layer
	 * @param wInit     (WeightInit) weight initialisation method
	 */
	public EmbeddingBag(int vocabSize, int outdims, WeightInit wInit) {
		// this.vocabSize=vocabSize;
		this.outdims = outdims;
		this.W = wInit.generate(vocabSize, outdims);
		this.gW = DoubleMatrix.zeros(vocabSize, outdims);
	}
	public EmbeddingBag(int vocabSize, int outdims,WeightInit wInit, A4Dataset vocabset) {
		this.outdims = outdims;
		this.gW = DoubleMatrix.zeros(vocabSize, outdims);
		this.vocabset=vocabset;
		this.W = DoubleMatrix.zeros(vocabSize, outdims);
		
		initW(); // assign vector values to weights
	}
	private DoubleMatrix initW() {
		ArrayList<double[]> items = this.vocabset.getvocabItems(); // store vector values in this ArrayList
		
		for(int i=0;i<items.size();i++) {
			double[] vector = items.get(i);
			for(int j=0;j<vector.length;j++) {
				// assign the pre-trained vector to W
				this.W.put(i,j,vector[j]);
			}
		}
		return null;
	}
	/**
	 * Forward pass
	 * 
	 * @param input (List<int[]>) input for forward calculation
	 * @return a [batchsize x outdims] matrix, each row is the output of a sample in
	 *         the batch
	 */
	@Override
	public DoubleMatrix forward(Object input) {
		// output of this layer (to be computed by you)
		// YOUR CODE HERE

		DoubleMatrix X = (DoubleMatrix) input;
		DoubleMatrix Y = DoubleMatrix.zeros(X.rows, W.columns);

		getX(input);

		// Y = X * W
			for(int i=0;i<this.X.size();i++) {
				// Iterate the samples in the batch
				for(int k=0;k<W.columns;k++) {
					// Iterate each W
					double sum=0;
					for(int j=0;j<this.X.get(i).length;j++) {
						// get the sum of W where x = 1
						int num = this.X.get(i)[j];
						sum=sum+W.get(num,k);
					}
					Y.put(i, k, sum);
				}
			}

		return Y;
	}

	
	@Override
	public DoubleMatrix backward(DoubleMatrix gY) {
		// YOUR CODE HERE

		
		for(int k=0;k<outdims;k++) {
			// Iterate the output nodes
			for(int i=0;i<X.size();i++) {
				// Iterate the samples in the batch
				for(int j=0;j<X.get(i).length;j++) {
					// update gW
					int num = this.X.get(i)[j];
					double gy = gY.get(i, k); // the value of current sample
					double gw = gW.get(num,k); // the previous gradient value
				
					gW.put(num, k, gy+gw);
				}
			}
		}

		return null; // there is no need to compute gX as the previous layer of this one is the input
						// layer of the network
	}
	private void getX(Object input) {
		// get the word indices from input object
		this.X = new ArrayList<>(); // (an array of word indices).

		DoubleMatrix X = (DoubleMatrix) input;
		for (int i = 0; i < X.rows; i++) {
			ArrayList<Integer> arr = new ArrayList<>();
			for (int j = 0; j < X.columns; j++) {
				if (X.get(i, j) > 0) {
					arr.add(j);
				}
			}
			int[] Original = new int[arr.size()];
			for (int m = 0; m < arr.size(); m++) {
				Original[m] = arr.get(m);
			}
			this.X.add(Original); /// add the original word indices into X List

		}
	}

	@Override
	public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
		weights.add(W);
		return weights;
	}

	@Override
	public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
		grads.add(gW);
		return grads;
	}

	@Override
	public String toString() {
		return String.format("Embedding: %d rows, %d dims", W.rows, W.columns);
	}

}
