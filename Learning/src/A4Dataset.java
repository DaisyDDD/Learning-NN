package src;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import minet.data.Dataset;
import minet.util.Pair;

public class A4Dataset extends Dataset<double[], Integer> {

	int inputDims; // number of input features
	protected ArrayList<double[]> vectorValues; // vector values of each word in part 3
	protected ArrayList<String> word; // all words in vocabulary file in part 3
	
	public A4Dataset(int batchsize, boolean shuffle, Random rnd) {
		super(batchsize, shuffle, rnd);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Get the number of input features (3249)
	 */
	public int getInputDims() {
		return inputDims;
	}

	/**
	 * Load A4 data from file.
	 */

	public void fromFile(String path) throws IOException {
		// Input data file:
		// Each following line: [input features (a list of double values, separated by
		// spaces)] ; [output label (an integer)]

		items = new ArrayList<Pair<double[], Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(path));
		String[] ss;
		
		while (true) {

			String line = br.readLine();
			if (line == null) {
				break;
			}
			int[] inputX = new int[inputDims];
			ss = line.split(" ; ");
			String[] sx = ss[0].split(" ");
			
			for (int j = 0; j < sx.length; j++) {
				// Transfer to one-hot-encoding format
				inputX[Integer.parseInt(sx[j])] = 1;
			}

			double[] xs = new double[inputDims];
			Integer y = Integer.valueOf(ss[1]);
			for (int j = 0; j < inputX.length; j++) {
				// change to double
				xs[j] = Double.valueOf(inputX[j]);
			}
			items.add(new Pair<double[], Integer>(xs, y));
		}
		br.close();
	}

	public void readVocab(String path) throws IOException {
		// Read vocabulary file in part 3
		// Store the words and relevant vector values
		vectorValues = new ArrayList<>();
		word = new ArrayList<>();
		
		BufferedReader br = new BufferedReader(new FileReader(path));
		items = new ArrayList<Pair<double[], Integer>>();
		String[] sx;
		inputDims = 100;
		
		while (true) {

			String line = br.readLine();
			if (line == null) {
				break;
			}
			sx = line.split(" ");
			
			//get each word 
			word.add(sx[0]); 
			double[] xs = new double[inputDims];
			
			for (int j = 1; j < sx.length; j++) {
				xs[j-1] = Double.parseDouble(sx[j]);
			}
			//get the vector of word
			vectorValues.add(xs);
		}
		
	}
	public void setInputDims(int inputDims) {
		this.inputDims=inputDims;
	}

	public ArrayList<double[]> getvocabItems() {
		return vectorValues;
	}

	public ArrayList<String> getWord() {
		return word;
	}

}
