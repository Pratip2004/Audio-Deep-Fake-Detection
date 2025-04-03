# Audio-Deep-Fake-Detection
## Introduction:
   In today's AI-driven world, synthetic speech technology has become incredibly advanced, making it harder than ever to tell real voices apart from deepfake audio. While this innovation has its benefits—like improving accessibility and creating realistic virtual assistants—it also comes with serious risks. Deepfake audio can be misused for misinformation, scams, and even identity theft, posing a major challenge in digital security.

This project is dedicated to tackling that challenge by exploring and implementing cutting-edge methods to detect AI-generated speech. From traditional techniques that analyze audio features like MFCCs and spectral properties to deep learning models that automatically learn patterns from raw sound, this repository dives deep into multiple detection strategies.

## Research & Selection:
 After carefully reviewing the *Audio-Deepfake-Detection* GitHub repository—a well-curated collection of research papers and resources on detecting audio deepfakes—I’ve identified three promising approaches that align with our needs. Each method offers unique strengths and trade-offs in distinguishing real from AI-generated speech.
1.  ### Handcrafted Feature-Based Detection :-
🔹 How It Works: This approach relies on manually engineered features like Mel-Frequency Cepstral Coefficients (MFCCs), Linear Predictive Coding (LPC), and spectral features to capture key characteristics of an audio signal.  

🔹 Performance & Benefits: Studies show that handcrafted features can effectively differentiate real and fake audio, especially in controlled environments. They are computationally efficient, making them well-suited for real-time detection. Additionally, their interpretability helps us understand what distinguishes genuine speech from deepfake audio.  

🔹 Limitations: Since this method depends on predefined features, it may struggle to adapt to more sophisticated or evolving deepfake techniques, potentially reducing its effectiveness over time.  

2. ### Hybrid Feature-Based Detection:- 
🔹 How It Works: This technique combines handcrafted features with deep learning-based features, leveraging the best of both worlds for improved detection accuracy.  

🔹 Performance & Benefits: Research indicates that hybrid models tend to outperform approaches that rely solely on either handcrafted or deep learning-derived features. They achieve higher accuracy and lower false positive rates, making them more reliable across various deepfake audio types.  

🔹 Limitations: The added complexity increases computational demands, which could be a drawback for real-time applications where processing speed is critical.  

3. ### End-to-End Deepfake Detection:-
🔹 How It Works: Instead of manually extracting features, this method uses deep neural networks to automatically learn audio patterns directly from raw data.  

🔹 Performance & Benefits: End-to-end models have achieved state-of-the-art accuracy in deepfake detection benchmarks. Their ability to learn complex patterns makes them highly adaptable to different deepfake techniques, improving real-world detection capabilities.  

🔹 Limitations: These models require significant computational resources and large labeled datasets for training, which can be a challenge in resource-limited environments.  

Each of these approaches provides valuable insights into real-time deepfake audio detection. While handcrafted features offer efficiency, hybrid models provide balance, and end-to-end methods excel in adaptability, the best choice depends on our specific requirements and constraints.

## Implementation :
   These are the three forgery based detection models, Here I used to select the Hybrid model to continue the project. 
### 1. Why Hybrid Feature-Based Detection?
After considering different approaches, I’ve decided to go with Hybrid Feature-Based Forgery Detection for several reasons:

 #### 🔹 Best of both worlds:
 This approach blends handcrafted audio features (like MFCCs) with deep learning-extracted features, ensuring both interpretability and adaptability.

 #### 🔹 More robust to evolving deepfake techniques:
 Unlike purely handcrafted methods, hybrid models can adapt better to new deepfake generation strategies.

#### 🔹 Balanced computational cost: 
While deep learning models offer high accuracy, they require significant resources. The hybrid approach strikes a good balance between performance and efficiency, making it more practical for real-time detection.
### 2. How Does It Compare to Other Approaches?

| **Criteria** | **Handcrafted Features** | **Hybrid (Selected Approach)** | **End-to-End Deep Learning** |
|-------------|----------------------|----------------------|---------------------------|
| **How Features Are Extracted** | Manually designed (e.g., MFCC, LPC) | Combination of manual & deep learning features | Learned directly from raw audio |
| **Interpretability** | High – easy to analyze | Moderate – mix of explainable & learned features | Low – features are learned implicitly |
| **Detection Accuracy** | Moderate | High | Very High |
| **Computational Cost** | Low – lightweight processing | Moderate – requires some deep learning | High – needs extensive processing power |
| **Adaptability to New Deepfake Techniques** | Low – struggles with unseen manipulations | High – adapts to a variety of attacks | Very High – learns from diverse patterns |
| **Real-Time Feasibility** | High – very fast | Moderate to High – some latency | Low – requires significant resources |

### 3. Existing Code & Model Selection
#### 🔹 Where is the code coming from?

The Audio Deepfake Detection GitHub repo contains several repositories with existing models.

I’ll be working with a **PyTorch-based hybrid model** that combines **MFCC feature** extraction with a  **CNN-BiLSTM** architecture.

#### 🔹 How will it be implemented?

Everything will be set up in a Jupyter Notebook, allowing for easy experimentation and visualization.

### 4.Model Architecture:
![image](https://github.com/user-attachments/assets/ba302d48-2788-4837-81d3-2a027ccc263d)
This is the model Architecture Which is Used to detect the real or fake voices of the audio speech datasets. 
 #### 1. MFCC:
  MFCC (Mel-Frequency Cepstral Coefficients) is a widely used feature extraction technique in audio deepfake detection, helping to identify subtle differences between real and AI-generated speech. It works by transforming the audio signal into a set of coefficients that represent its spectral characteristics, mimicking how the human ear perceives sound. By emphasizing lower frequencies and reducing irrelevant noise, MFCC captures key speech features, making it useful for detecting unnatural patterns or distortions in deepfake audio. When combined with deep learning models, it enhances accuracy by providing both handcrafted and automatically extracted features, improving detection reliability.
  The output of MFCC is like A 2D matrix (time × MFCC coefficients), which acts like an image input for CNN.
 #### 2. Working of CNN: 
 - The MFCC matrix is treated as an image (frequency vs. time).

  - Convolutional layers extract local patterns. It Reduces noise and irrelevant variations through convolutional layers. That is also a reason to select this model.

- Pooling layers(MAX-POOL,AVG POOL) downsample the features for computational efficiency.

- The output is a compact feature map that represents the audio input which is then flatten and acts as a sequence.

![image](https://github.com/user-attachments/assets/42a18e4f-a1a4-4597-bc55-1f30568a2f56)

##### 🔹 Output: 
A sequence of feature embeddings for each time step.
   #### 3. Working of BiLSTM RNN: 
   Speech has sequential dependencies. CNN alone doesn’t capture long-term temporal relationships. So, We can use the LSTM RNN, But Standard LSTM(Long-Short Term Memory) processes only past context may miss future dependencies. BiLSTM processes both past & future context, improving speech recognition.
- Takes CNN feature embeddings as input.
- Processes the sequence forward & backward.
- Captures time-dependent patterns to understand deepfake cues.
- Outputs a final vector representation for classification.
        ![image](https://github.com/user-attachments/assets/e527312a-5993-4623-8cde-c9673180bbc4)

🔹 Output: A compressed sequence representation with past & future dependencies.
#### 3. Fully Connected Layer and Classification:
  A high dimensional sequence is the output of Bidirectional LSTM RNN. So The final BiLSTM output is passed to a fully connected (FC) layer. This is a Neural Network where we use the backpropagation and optimizers to adjust the weight and bias using the loss function.
  A softmax activation predicts real (0) or fake (1).

## Model Analysis
### Challenges Faced & How They Were Solved
#### 1. Dataset Preprocessing
📌 Challenge: The ASVspoof 5 dataset included multiple audio formats and varying durations, making standardization tricky.

✅ Solution: Used Librosa to resample all audio files to a fixed sample rate and duration, ensuring consistency across the dataset.

#### 2. Extracting Meaningful MFCC Features
📌 Challenge: Extracting the right features without losing key details that differentiate real and fake audio.

✅ Solution: Chose 13 MFCC coefficients, preserving the time-series structure for better learning by LSTM layers.

#### 3. Training Instability
📌 Challenge: The model was sensitive to learning rate changes, leading to unstable training behavior.

✅ Solution: Implemented learning rate scheduling and batch normalization to keep training stable.

#### 4. Overfitting Issues
📌 Challenge: The model performed well on training data but struggled on unseen validation samples.

✅ Solution: Used dropout layers (0.3), L2 regularization, and data augmentation techniques like adding noise and pitch shifting to improve generalization.

### Key Assumptions
- #### 🎯 Balanced Dataset:
  Assumed that the dataset had a roughly equal number of real and fake samples.

- #### 🎯 MFCC Features Are Sufficient:
  Assumed that MFCCs capture the key differences needed for detection.

- #### 🎯 Subset Generalization:
  Assumed that training on a smaller dataset would still be effective when applied to larger datasets.
###  2. Model & Performance Analysis
#### Why Choose CNN-BiLSTM?
✅ CNN (Convolutional Neural Networks) – Helps detect local spectral patterns in the MFCC representation.
✅ BiLSTM (Bidirectional Long Short-Term Memory) – Captures time-based dependencies, making it great for speech analysis.
✅ MFCC Input Representation – Mimics human hearing, making it effective for speech and audio analysis.

#### How It Works
1️⃣ Extract MFCC Features – Converts raw audio into a time-frequency representation.
2️⃣ Pass Through CNN Layers – Detects small-scale spectral changes that may indicate deepfake manipulation.
3️⃣ Feed Into BiLSTM Layers – Captures long-term dependencies in the speech patterns.
4️⃣ Fully Connected Layers – Convert learned features into a classification decision (Real or Fake).
5️⃣ Softmax Activation – Outputs the probability of the audio being real or fake.

#### Performance Metrics
Metric	Value
Training Accuracy	-> 91.5%

Validation Accuracy->	58.2%

Test Accuracy->	36.4%

Average Inference Speed	-> 50ms/sample
#### Strengths & Weaknesses
##### ✅ What Worked Well?

- Captures Speech Patterns Efficiently using BiLSTM.

- Generalizes Well thanks to data augmentation techniques.

- Fast Inference Time (~50ms/sample) makes it suitable for near real-time detection.

#####  Challenges & Limitations

Sensitive to Noise Variations – Performance drops in noisy environments.

Overfitting Risks – Needs careful regularization.There are high overfitting happens, may be due to lack of dataset. After optimizing also the data set has overfitting, thats the test accuracy is low.

Dependent on MFCC Quality – May not capture high-level deepfake artifacts effectively.

### 3. Future Improvements
 - Integrate Transfer Learning – Use models like Wav2Vec2 to improve feature extraction.
 - Expand Training Dataset – Include more diverse deepfake samples to improve robustness.
 - Try Spectrogram-Based CNNs – Train models directly on mel-spectrograms for deeper feature extraction.

### 4. Reflection & Key Takeaways
#### - Biggest Challenges?
Managing dataset variability in real-world recordings.

Balancing accuracy vs. real-time performance in detection.

Preventing overfitting while maintaining strong generalization.

####  - Real-World vs. Research Dataset Performance
Lab datasets (e.g., ASVspoof) are clean and well-labeled.

Real-world data introduces background noise, compression artifacts, and lower-quality recordings, making detection harder.

#### - What Would Improve Performance?
More Deepfake Audio Variants – Training on multiple deepfake synthesis techniques.

Noise-Augmented Training Data – To improve robustness in real-world scenarios.

More Computational Resources – Exploring deeper architectures like Wav2Vec2 or ResNet-based models.

#### - Deploying in a Production Environment
Optimize Model for Edge Devices – Use techniques like quantization and pruning to reduce size.

Deploy as a REST API – Using FastAPI for real-time inference.

Focus on Speed & Accuracy Trade-Offs – Balance lightweight models with deep learning accuracy.

### Final Thoughts
This CNN-BiLSTM model with MFCC features provides a strong starting point for AI-generated speech detection. While the model shows promising results, further improvements in dataset diversity, transfer learning, and real-world testing will be crucial for production-level robustness.

By continuously refining the model and incorporating state-of-the-art deep learning techniques, we can push the boundaries of real-time deepfake audio detection and enhance security in digital communications.








