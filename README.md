# Intelligent Flappy Bird Agent Using Deep Q-Learning

## Team Information
This project was developed by a two-person team, both students in the Master's Program - Artificial Intelligence and Optimization at “Alexandru Ioan Cuza" University. The team members are:
- **Burcă Theodor**  
- **Duluță George-Denis**  

## Project Overview
This report outlines the architecture and methodology used to train an intelligent Deep Q-Learning (DQL) agent for the `FlappyBird-v0` environment. The project leverages advanced reinforcement learning techniques to optimize performance and achieve competitive results.

## Architecture Details
The Deep Q-Network (DQN) employed in this project is designed to handle the complex environment of Flappy Bird efficiently. The network begins with an input layer matching the environment's observation space dimensions. This is followed by a fully connected layer consisting of 512 nodes activated via ReLU.

An optional Dueling DQN architecture is incorporated for enhanced performance. This architecture includes a value stream that computes the state value and an advantages stream for action-based values. The Q-value is calculated as:

$`Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a'))`$

The output layer maps the hidden states to action Q-values, directly in non-dueling cases.

Replay memory is utilized for efficient training with a size of up to 100,000 transitions. This memory randomly samples mini-batches to avoid overfitting and ensure diversity.

The training process is implemented in PyTorch, using Mean Squared Error (MSE) as the loss function and Adam optimizer with a learning rate of 0.0001. An epsilon-greedy strategy is employed to balance exploration and exploitation.

## Hyperparameters
All hyperparameters are loaded from a YAML configuration file. The key parameters are as follows:

- Environment ID: FlappyBird-v0.
- Replay Memory Size: 100,000.
- Mini-batch Size: 32.
- Epsilon Decay Rate: 0.99995.
- Minimum Epsilon: 0.05.
- Network Sync Rate: Every 10 steps.
- Learning Rate ($`\alpha`$): 0.0001.
- Discount Factor ($`\gamma`$): 0.99
- Hidden Nodes (FC1): 512.
- Enable Double DQN: True.
- Enable Dueling DQN: True.

## Experimentation
Training the agent took approximately 14 hours, and the highest score achieved during the training process was 1674. The training involved running multiple episodes with the FlappyBird-v0 environment. The agent's performance gradually improved as the epsilon value decayed, encouraging more exploitation over time. The best-performing model was saved based on the highest reward obtained during the training sessions.
Throughout the experimentation phase, various hyperparameters such as the learning rate, epsilon decay rate, and network architecture were adjusted to find the optimal configuration. This iterative process of fine-tuning hyperparameters contributed significantly to enhancing the agent’s performance.

## Observations and Challenges
Epsilon decay posed a critical challenge, with fast decay rates often leading to premature exploitation. Adjusting decay rates was essential to balance exploration and exploitation.

While CUDA acceleration significantly boosted performance, transferring small data batches occasionally caused the CPU to outperform the GPU.

Reward variance during early training episodes was successfully reduced by using Double DQN, mitigating overestimation of Q-values.

## Future Work
Future enhancements to the project could focus on improving the model's input processing to achieve higher performance scores. Specifically:

- Implementing feature extraction techniques (e.g., using sensors or simulating LiDAR for the bird) to preprocess images and provide the model with more complex, informative features. This approach aligns with the 25-point scoring criteria and could enhance the agent’s ability to generalize in the game.

- Training the model directly on the raw pixel data of the game frames. This would involve pre-processing steps such as resizing, grayscale conversion, thresholding, dilation, erosion, or background removal to make the input data more suitable for the network. Meeting the 30-point scoring criteria through this method would fully leverage the visual data for decision-making.

These approaches could significantly improve the model's performance and robustness in more complex scenarios.

## Video
https://drive.google.com/file/d/1PT1l4jNqPF17lLPhRY9x8xSrF6dUEJep/view?usp=sharing

## References
1. Mnih et al., "Playing Atari with Deep Reinforcement Learning," 2013.
2. van Hasselt et al., "Deep Reinforcement Learning with Double Q-Learning," 2015.
3. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," 2016.
4. https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi
